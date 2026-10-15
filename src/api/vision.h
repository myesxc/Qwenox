#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer.h"

namespace vision {

using json = nlohmann::ordered_json;

struct Frame {
    std::array<int, 3> grid{1, 0, 0};
    std::vector<float> patches;

    int pad_tokens() const { return grid[0] * grid[1] * grid[2] / 4; }
};

// Decode a data URL or bare base64 payload. HTTP(S) is deliberately rejected.
bool decode_image_url(const std::string& value, std::vector<uint8_t>* bytes,
                      std::string* error);

// Per-request image cap: QWENOX_API_MAX_IMAGES (service.conf MAX_IMAGES),
// default 8, ceiling 256 (= engine GEN-protocol frame ceiling).
int max_images();

// Added-token ids that only mean "here is media" inside a placeholder span the
// chat template placed (<|vision_start|><|image_pad|><|vision_end|> and the
// video/audio siblings). A literal mention of one of these strings anywhere
// else — a user paste, a tool result, the model's own earlier answer — must
// encode as ordinary text: as an added token the engine reads it as vision
// input it has no patches for, and the model, which never saw the sequence
// outside a template span, tends to close its turn right after writing it.
// Also used to keep the token cache from reusing a span whose text this
// request would encode differently.
const std::unordered_set<int>& placeholder_marker_ids();

// Decode PNG/JPEG/WebP, smart-resize it, normalize it, and build Qwen patches.
bool preprocess(const std::vector<uint8_t>& bytes, Frame* frame, std::string* error);

// Extract and preprocess image parts from normalized chat messages, in order.
bool prepare_messages(const json& messages, std::vector<Frame>* frames,
                      std::string* error);

// Expand only the image-pad tokens the chat template placed. `pad_offsets`
// (ascending) lists the byte offset of <|image_pad|> inside every placeholder
// span the render inserted (RenderResult::image_pad_offsets); only those
// positions are placeholders. Any other <|image_pad|>/<|video_pad|> token in
// the text — a user-quoted span, a tool result carrying the literal — is
// re-encoded as ordinary text, never as vision. `starts` (optional) receives
// each id's byte offset in `prompt` (the copies of an expanded pad share the
// pad's offset). Only frames[i].pad_tokens() is read, so the frames may carry
// grids without patches.
bool encode_prompt(const std::string& prompt, const qwenox::Tokenizer& tokenizer,
                   const std::vector<Frame>& frames, const std::vector<size_t>& pad_offsets,
                   std::vector<int>* ids, std::string* error,
                   std::vector<size_t>* starts = nullptr);

}  // namespace vision
