#include "scan.h"

void scan(std::vector<float>& input, std::vector<float>& output) {
    if (input.empty()) return;

    output[0] = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i];
    }
}
