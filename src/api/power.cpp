#include "power.h"

#include <mutex>

#ifdef _WIN32

#include <cstdlib>
#include <cstring>

#include <windows.h>

namespace {

// ADL PMLog 结构（ADL SDK adl_structures.h / adl_defines.h）：
// ADL2_New_QueryPMLogData_Get 返回全部传感器快照，按 ADL_PMLOG_SENSORS
// 枚举直接索引。
constexpr int ADL_PMLOG_MAX_SENSORS = 256;
constexpr int ADL_PMLOG_ASIC_POWER = 23;          // W，整封装
constexpr int ADL_PMLOG_GFX_POWER = 30;           // W，GPU 域
constexpr int ADL_PMLOG_CPU_POWER = 33;           // W，CPU 域
constexpr int ADL_PMLOG_SSPAIRED_ASICPOWER = 46;  // W，APU 整封装
constexpr int ADL_PMLOG_BOARD_POWER = 73;         // W，板卡
constexpr int ADL_PMLOG_TEMPERATURE_EDGE = 8;     // °C，GPU 边缘
constexpr int ADL_PMLOG_TEMPERATURE_HOTSPOT = 27; // °C，GPU 热点
constexpr int ADL_PMLOG_TEMPERATURE_GFX = 28;     // °C，GPU 核心
constexpr int ADL_PMLOG_TEMPERATURE_SOC = 29;     // °C，SoC
constexpr int ADL_PMLOG_TEMPERATURE_CPU = 32;     // °C，CPU

struct ADLSingleSensorData {
    int supported;
    int value;
};
struct ADLPMLogDataOutput {
    int size;
    ADLSingleSensorData sensors[ADL_PMLOG_MAX_SENSORS];
};  // 4 + 256*8 = 2052 字节

typedef void*(__stdcall* ADL_MALLOC_CB)(int);
typedef int(__stdcall* PFN_CREATE2)(ADL_MALLOC_CB, int, void**);
typedef int(__stdcall* PFN_PMLOG_QUERY)(void*, int, ADLPMLogDataOutput*);

void* __stdcall adl_malloc(int size) { return malloc(static_cast<size_t>(size)); }

// 进程级单例，刻意不析构（进程退出时系统回收，避免 at-exit 与在途查询竞争）。
HMODULE g_dll = nullptr;
void* g_ctx = nullptr;  // ADL_CONTEXT_HANDLE
PFN_PMLOG_QUERY g_query = nullptr;
int g_adapter = -1;     // 选定的适配器索引（第一个能读出功耗的）
int g_state = 0;        // 0 未初始化 / 1 可用 / -1 不可用
std::mutex g_mtx;

void adl_init() {
    g_dll = LoadLibraryA("atiadlxx.dll");
    if (!g_dll) g_dll = LoadLibraryA("atiadlxy.dll");
    if (!g_dll) {
        g_state = -1;
        return;
    }
    PFN_CREATE2 create2 =
        reinterpret_cast<PFN_CREATE2>(GetProcAddress(g_dll, "ADL2_Main_Control_Create"));
    g_query =
        reinterpret_cast<PFN_PMLOG_QUERY>(GetProcAddress(g_dll, "ADL2_New_QueryPMLogData_Get"));
    if (!create2 || !g_query || create2(adl_malloc, 1, &g_ctx) != 0 || !g_ctx) {
        g_state = -1;
        return;
    }
    // 探测适配器：取第一个 PMLog 查询成功且支持任一功耗或温度传感器的。
    for (int i = 0; i < 8; ++i) {
        ADLPMLogDataOutput pm;
        memset(&pm, 0, sizeof(pm));
        pm.size = static_cast<int>(sizeof(pm));
        if (g_query(g_ctx, i, &pm) != 0) continue;
        if (pm.sensors[ADL_PMLOG_ASIC_POWER].supported ||
            pm.sensors[ADL_PMLOG_GFX_POWER].supported ||
            pm.sensors[ADL_PMLOG_SSPAIRED_ASICPOWER].supported ||
            pm.sensors[ADL_PMLOG_TEMPERATURE_EDGE].supported ||
            pm.sensors[ADL_PMLOG_TEMPERATURE_GFX].supported ||
            pm.sensors[ADL_PMLOG_TEMPERATURE_CPU].supported) {
            g_adapter = i;
            g_state = 1;
            return;
        }
    }
    g_state = -1;
}

int sensor_value(const ADLPMLogDataOutput& pm, int sensor) {
    return pm.sensors[sensor].supported ? pm.sensors[sensor].value : -1;
}

// 温度：部分驱动对 APU 不存在的传感器也报 supported 但值为 0，只收 1..150 °C。
int temp_value(const ADLPMLogDataOutput& pm, int sensor) {
    const int v = sensor_value(pm, sensor);
    return (v >= 1 && v <= 150) ? v : -1;
}

}  // namespace

namespace power {

Reading read() {
    Reading out;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_state == 0) adl_init();
    if (g_state != 1) return out;
    ADLPMLogDataOutput pm;
    memset(&pm, 0, sizeof(pm));
    pm.size = static_cast<int>(sizeof(pm));
    if (g_query(g_ctx, g_adapter, &pm) != 0) return out;
    out.socket_watts = sensor_value(pm, ADL_PMLOG_ASIC_POWER);
    if (out.socket_watts < 0) out.socket_watts = sensor_value(pm, ADL_PMLOG_SSPAIRED_ASICPOWER);
    if (out.socket_watts < 0) out.socket_watts = sensor_value(pm, ADL_PMLOG_BOARD_POWER);
    out.gfx_watts = sensor_value(pm, ADL_PMLOG_GFX_POWER);
    out.cpu_watts = sensor_value(pm, ADL_PMLOG_CPU_POWER);
    out.available =
        out.socket_watts >= 0 || out.gfx_watts >= 0 || out.cpu_watts >= 0;
    out.gpu_temp_c = temp_value(pm, ADL_PMLOG_TEMPERATURE_GFX);
    if (out.gpu_temp_c < 0) out.gpu_temp_c = temp_value(pm, ADL_PMLOG_TEMPERATURE_EDGE);
    out.gpu_hotspot_c = temp_value(pm, ADL_PMLOG_TEMPERATURE_HOTSPOT);
    out.cpu_temp_c = temp_value(pm, ADL_PMLOG_TEMPERATURE_CPU);
    out.soc_temp_c = temp_value(pm, ADL_PMLOG_TEMPERATURE_SOC);
    out.temp_available = out.gpu_temp_c >= 0 || out.gpu_hotspot_c >= 0 ||
                         out.cpu_temp_c >= 0 || out.soc_temp_c >= 0;
    return out;
}

}  // namespace power

#elif defined(__linux__)

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>

#include <dirent.h>
#include <strings.h>

namespace {

// 路径在首次调用时解析一次，之后每次只读文件（sysfs 读取是纯内核调用）。
// 功耗: /sys/class/drm/cardN/device/hwmon/hwmonX/power1_average（µW）。
// GPU 温度: 同一 amdgpu hwmon 的 tempN_input（m°C），按 tempN_label 认
//   "edge" / "junction"；没有 label 时 temp1 当 edge。
// CPU 温度: /sys/class/hwmon/hwmonX，name = k10temp（或 zenpower），
//   取 label 为 Tctl/Tdie 的那路，否则 temp1。
std::string g_power_path, g_gpu_edge_path, g_gpu_junc_path, g_cpu_path;
bool g_inited = false;
std::mutex g_mtx;

bool file_exists(const std::string& p) {
    FILE* f = fopen(p.c_str(), "r");
    if (!f) return false;
    fclose(f);
    return true;
}

std::string read_line(const std::string& p) {
    FILE* f = fopen(p.c_str(), "r");
    if (!f) return {};
    char buf[128] = {0};
    if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
    fclose(f);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

// 在 hwmon 目录里找 label 等于 want 之一的 tempN_input；找不到返回空。
std::string find_temp_by_label(const std::string& base,
                               std::initializer_list<const char*> want) {
    for (int i = 1; i <= 16; ++i) {
        const std::string pre = base + "/temp" + std::to_string(i);
        const std::string label = read_line(pre + "_label");
        if (label.empty()) continue;
        for (const char* w : want)
            if (strcasecmp(label.c_str(), w) == 0 && file_exists(pre + "_input"))
                return pre + "_input";
    }
    return {};
}

void init_amdgpu() {
    for (int card = 0; card < 8; ++card) {
        const std::string dev =
            "/sys/class/drm/card" + std::to_string(card) + "/device";
        FILE* f = fopen((dev + "/vendor").c_str(), "r");
        if (!f) continue;
        unsigned int vendor = 0;
        const int n = fscanf(f, "0x%x", &vendor);
        fclose(f);
        if (n != 1 || vendor != 0x1002) continue;
        const std::string hwmon_dir = dev + "/hwmon";
        DIR* d = opendir(hwmon_dir.c_str());
        if (!d) continue;
        while (dirent* de = readdir(d)) {
            if (strncmp(de->d_name, "hwmon", 5) != 0) continue;
            const std::string base = hwmon_dir + "/" + de->d_name;
            if (g_power_path.empty()) {
                for (const char* name : {"power1_average", "power1_input"}) {
                    if (file_exists(base + "/" + name)) {
                        g_power_path = base + "/" + name;
                        break;
                    }
                }
            }
            if (g_gpu_edge_path.empty()) {
                g_gpu_edge_path = find_temp_by_label(base, {"edge"});
                if (g_gpu_edge_path.empty() && file_exists(base + "/temp1_input"))
                    g_gpu_edge_path = base + "/temp1_input";
            }
            if (g_gpu_junc_path.empty())
                g_gpu_junc_path = find_temp_by_label(base, {"junction", "hotspot"});
        }
        closedir(d);
        if (!g_power_path.empty() || !g_gpu_edge_path.empty()) return;
    }
}

void init_cpu() {
    DIR* d = opendir("/sys/class/hwmon");
    if (!d) return;
    std::string fallback;
    while (dirent* de = readdir(d)) {
        if (strncmp(de->d_name, "hwmon", 5) != 0) continue;
        const std::string base = std::string("/sys/class/hwmon/") + de->d_name;
        const std::string name = read_line(base + "/name");
        if (name != "k10temp" && name != "zenpower") continue;
        std::string p = find_temp_by_label(base, {"Tctl", "Tdie"});
        if (p.empty() && file_exists(base + "/temp1_input")) p = base + "/temp1_input";
        if (p.empty()) continue;
        if (name == "k10temp") {
            g_cpu_path = p;
            break;
        }
        if (fallback.empty()) fallback = p;
    }
    closedir(d);
    if (g_cpu_path.empty()) g_cpu_path = fallback;
}

// 读一个整数文件；失败返回 -1。
long long read_ll(const std::string& p) {
    if (p.empty()) return -1;
    FILE* f = fopen(p.c_str(), "r");
    if (!f) return -1;
    long long v = -1;
    if (fscanf(f, "%lld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

// m°C → °C（四舍五入），只收 1..150 °C。
int milli_to_c(long long mc) {
    if (mc < 0) return -1;
    const int c = static_cast<int>((mc + 500) / 1000);
    return (c >= 1 && c <= 150) ? c : -1;
}

}  // namespace

namespace power {

Reading read() {
    Reading out;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!g_inited) {
        init_amdgpu();
        init_cpu();
        g_inited = true;
    }
    const long long uw = read_ll(g_power_path);
    if (uw >= 0) {
        out.socket_watts = static_cast<int>(uw / 1000000);  // µW → W
        out.available = true;
    }
    out.gpu_temp_c = milli_to_c(read_ll(g_gpu_edge_path));
    out.gpu_hotspot_c = milli_to_c(read_ll(g_gpu_junc_path));
    out.cpu_temp_c = milli_to_c(read_ll(g_cpu_path));
    out.temp_available = out.gpu_temp_c >= 0 || out.gpu_hotspot_c >= 0 ||
                         out.cpu_temp_c >= 0;
    return out;
}

}  // namespace power

#else

namespace power {

Reading read() { return Reading{}; }

}  // namespace power

#endif
