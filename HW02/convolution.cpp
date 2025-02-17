//This task has been done with the help of chatgpt, claude.ai and google search
#include "convolution.h"
#include <vector>

// Helper function to handle boundary padding
float get_padded_value(const std::vector<float>& image, int n, int i, int j) {
    if (i >= 0 && i < n && j >= 0 && j < n) {
        return image[i * n + j]; // Inside the valid range
    } else if ((i >= 0 && i < n) || (j >= 0 && j < n)) {
        return 1.0f; // Edge padding
    } else {
        return 0.0f; // Corner padding
    }
}

void convolve(const std::vector<float>& image, std::vector<float>& output, 
              const std::vector<float>& mask, int n, int m) {
    int offset = m / 2; // Since m is always odd

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float sum = 0.0f;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    int img_x = x + i - offset;
                    int img_y = y + j - offset;

                    float img_value = get_padded_value(image, n, img_y, img_x);
                    sum += mask[i * m + j] * img_value;
                }
            }

            output[y * n + x] = sum;
        }
    }
}
 
