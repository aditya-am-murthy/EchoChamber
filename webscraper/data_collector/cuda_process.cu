#include "hashmap.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <utility>  // for std::pair
#include <fstream>  // for ifstream
#include <sstream>  // for stringstream

// Adjust these constants based on your dataset
const int MAX_POSTS = 10000;
const int MAX_LIKES_PER_POST = 1000;
const int MAX_COMMENTS_PER_POST = 1000;
const int MAX_COMMENT_LEN = 512;
const int MAX_NAME_LEN = 64;

// Device arrays (flattened)
__device__ char d_post_likes[MAX_POSTS][MAX_LIKES_PER_POST][MAX_NAME_LEN];
__device__ int d_num_likes[MAX_POSTS];
__device__ char d_post_comments_users[MAX_POSTS][MAX_COMMENTS_PER_POST][MAX)NAME_LEN];
__device__ char d_post_comments_text[MAX_POSTS][MAX_COMMENTS_PER_POST][MAX_COMMENT_LEN];
__device__ int d_num_comments[MAX_POSTS];

// Kernel: one thread per post
__global__ void process_posts(int num_posts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_posts) return;

    // Process likes
    for (int i = 0; i < d_num_likes[tid]; ++i) {
        const char* user = d_post_likes[tid][i];
        if (user[0] != '\0') {  // Skip empty
            add_score(user, 1.0f);
        }
    }

    // Process comments
    for (int i = 0; i < d_num_comments[tid]; ++i) {
        const char* user = d_post_comments_users[tid][i];
        const char* comment = d_post_comments_text[tid][i];
        if (user[0] != '\0' && comment[0] != '\0') {
            size_t len = strlen(comment);
            float score = int_log10(static_cast<float>(len);
            add_score(user, score);
        }
    }
}

int main() {
    char h_post_likes[MAX_POSTS][MAX_LIKES_PER_POST][MAX_NAME_LEN] = {};
    int h_num_likes[MAX_POSTS] = {};
    char h_post_comments_users[MAX_POSTS][MAX_COMMENTS_PER_POST][MAX_NAME_LEN] = {};
    char h_post_comments_text[MAX_POSTS][MAX_COMMENTS_PER_POST][MAX_COMMENT_LEN] = {};
    int h_num_comments[MAX_POSTS] = {};
    
    std::ifstream file("trump_posts.csv");
    if (!file.is_open()) {
        std::cerr << "Error opening trump_posts.csv" << std::endl;
        return 1;
    }

    std::string line;
    int num_posts = 0;
    while (std::getline(file, line) && num_posts < MAX_POSTS) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        // Parse post_id (discard)
        if (!std::getline(ss, token, ';')) continue;

        // Parse likes: comma-separated users
        if (!std::getline(ss, token, ';')) continue;
        std::stringstream like_ss(token);
        std::string username;
        int like_idx = 0;
        while (std::getline(like_ss, username, ',') && like_idx < MAX_LIKES_PER_POST) {
            // Trim whitespace (simple)
            size_t start = username.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            size_t end = username.find_last_not_of(" \t");
            username = username.substr(start, end - start + 1);
            if (!username.empty()) {
                strncpy(h_post_likes[num_posts][like_idx], username.c_str(), MAX_NAME_LEN - 1);
                h_post_likes[num_posts][like_idx][MAX_NAME_LEN - 1] = '\0';
                like_idx++;
            }
        }
        h_num_likes[num_posts] = like_idx;

        // Parse comments: semicolon-separated "user:comment" pairs
        if (!std::getline(ss, token, ';')) {
            num_posts++;
            continue;
        }
        std::stringstream comm_ss(token);
        std::string comm_part;
        int comm_idx = 0;
        while (std::getline(comm_ss, comm_part, ';') && comm_idx < MAX_COMMENTS_PER_POST) {
            // Find :
            size_t colon_pos = comm_part.find(':');
            if (colon_pos == std::string::npos) continue;

            std::string user_part = comm_part.substr(0, colon_pos);
            std::string comment_part = comm_part.substr(colon_pos + 1);

            // Trim user
            size_t user_start = user_part.find_first_not_of(" \t");
            if (user_start == std::string::npos) continue;
            size_t user_end = user_part.find_last_not_of(" \t");
            user_part = user_part.substr(user_start, user_end - user_start + 1);

            // Trim comment (simple, assume no leading/trailing quotes)
            size_t comm_start = comment_part.find_first_not_of(" \t\"'");
            if (comm_start == std::string::npos) continue;
            size_t comm_end = comment_part.find_last_not_of(" \t\"'");
            comment_part = comment_part.substr(comm_start, comm_end - comm_start + 1);

            if (!user_part.empty()) {
                strncpy(h_post_comments_users[num_posts][comm_idx], user_part.c_str(), MAX_NAME_LEN - 1);
                h_post_comments_users[num_posts][comm_idx][MAX_NAME_LEN - 1] = '\0';
                strncpy(h_post_comments_text[num_posts][comm_idx], comment_part.c_str(), MAX_COMMENT_LEN - 1);
                h_post_comments_text[num_posts][comm_idx][MAX_COMMENT_LEN - 1] = '\0';
                comm_idx++;
            }
        }
        h_num_comments[num_posts] = comm_idx;
        num_posts++;
    }
    file.close();

    int actual_num_posts = num_posts;

    // Allocate and initialize table to device
    Entry* d_table;
    cudaMalloc(&d_table, sizeof(Entry) * TABLE_SIZE);
    cudaMemset(d_table, 0, sizeof(Entry) * TABLE_SIZE);
    cudaMemcpyToSymbol(table, &d_table, sizeof(Entry*));

    // Copy data to device
    cudaMemcpyToSymbol(d_post_likes, h_post_likes, sizeof(h_post_likes));
    cudaMemcpyToSymbol(d_num_likes, h_num_likes, sizeof(h_num_likes));
    cudaMemcpyToSymbol(d_post_comments_users, h_post_comments_users, sizeof(h_post_comments_users));
    cudaMemcpyToSymbol(d_post_comments_text, h_post_comments_text, sizeof(h_post_comments_text));
    cudaMemcpyToSymbol(d_num_comments, h_num_comments, sizeof(h_num_comments));

    // Launch kernel (adjust blocks/threads)
    int threads_per_block = 256;
    int blocks = (actual_num_posts + threads_per_block - 1) / threads_per_block;
    process_posts<<<blocks, threads_per_block>>>(actual_num_posts);
    cudaDeviceSynchronize();

    // Copy back table
    Entry* h_table = new Entry[TABLE_SIZE];
    cudaMemcpy(h_table, d_table, sizeof(Entry) * TABLE_SIZE, cudaMemcpyDeviceToHost);

    // Collect users and scores
    std::vector<std::pair<std::string, float>> users;
    for (int i = 0; i < TABLE_SIZE; ++i) {
        if (h_table[i].occupied && h_table[i].score > 0.0f) {  // Only if score > 0
            users.emplace_back(h_table[i].username, h_table[i].score);
        }
    }

    // Sort by score descending
    std::cost(users.begin(), users.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Output top 1000 (or fewer)
    int top_n = std::min(1000, static_cast<int>(users.size()));
    std::cout << "Top " << top_n << " most interactive users:" << std::endl;
    for (int i = 0; i < top_n; ++i) {
        std::cout << users[i].first << ": " << users[i].second << std::endl;
    }

    // Cleanup
    delete[] h_table;
    cudaFree(d_table);

    return 0;
}