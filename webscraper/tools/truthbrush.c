#include "truthbrush.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 1024

/* Helper function to execute a command and capture output */
static char* run_command(const char* command) {
    FILE* fp = popen(command, "r");
    if (!fp) {
        return NULL;
    }

    char* output = NULL;
    size_t output_size = 0;
    char buffer[BUFFER_SIZE];

    /* Read output incrementally */
    while (fgets(buffer, BUFFER_SIZE, fp) != NULL) {
        size_t len = strlen(buffer);
        char* temp = realloc(output, output_size + len + 1);
        if (!temp) {
            free(output);
            pclose(fp);
            return NULL;
        }
        output = temp;
        strcpy(output + output_size, buffer);
        output_size += len;
    }

    pclose(fp);
    return output;
}

Truthbrush* truthbrush_init(const char* truthbrush_path) {
    Truthbrush* tb = malloc(sizeof(Truthbrush));
    if (!tb) {
        return NULL;
    }
    tb->truthbrush_path = strdup(truthbrush_path ? truthbrush_path : "truthbrush");
    if (!tb->truthbrush_path) {
        free(tb);
        return NULL;
    }
    return tb;
}

void truthbrush_free(Truthbrush* tb) {
    if (tb) {
        free(tb->truthbrush_path);
        free(tb);
    }
}

char* truthbrush_search(Truthbrush* tb, const char* search_type, const char* query) {
    if (!tb || !search_type || !query) return NULL;
    if (strcmp(search_type, "accounts") != 0 && strcmp(search_type, "statuses") != 0 &&
        strcmp(search_type, "hashtags") != 0 && strcmp(search_type, "groups") != 0) {
        return NULL;
    }

    char command[512];
    snprintf(command, sizeof(command), "%s search --searchtype %s \"%s\"",
             tb->truthbrush_path, search_type, query);
    return run_command(command);
}

char* truthbrush_statuses(Truthbrush* tb, const char* handle) {
    if (!tb || !handle) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s statuses \"%s\"",
             tb->truthbrush_path, handle);
    return run_command(command);
}

char* truthbrush_suggestions(Truthbrush* tb) {
    if (!tb) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s suggestions", tb->truthbrush_path);
    return run_command(command);
}

char* truthbrush_tags(Truthbrush* tb) {
    if (!tb) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s tags", tb->truthbrush_path);
    return run_command(command);
}

char* truthbrush_ads(Truthbrush* tb) {
    if (!tb) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s ads", tb->truthbrush_path);
    return run_command(command);
}

char* truthbrush_user(Truthbrush* tb, const char* handle) {
    if (!tb || !handle) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s user \"%s\"",
             tb->truthbrush_path, handle);
    return run_command(command);
}

char* truthbrush_likes(Truthbrush* tb, const char* post_id, int top_num) {
    if (!tb || !post_id) return NULL;
    char command[512];
    if (top_num > 0) {
        snprintf(command, sizeof(command), "%s likes \"%s\" --includeall --top_num %d",
                 tb->truthbrush_path, post_id, top_num);
    } else {
        snprintf(command, sizeof(command), "%s likes \"%s\" --includeall",
                 tb->truthbrush_path, post_id);
    }
    return run_command(command);
}

char* truthbrush_comments(Truthbrush* tb, const char* post_id, int top_num) {
    if (!tb || !post_id) return NULL;
    char command[512];
    if (top_num > 0) {
        snprintf(command, sizeof(command), "%s comments \"%s\" --includeall --onlyfirst --top_num %d",
                 tb->truthbrush_path, post_id, top_num);
    } else {
        snprintf(command, sizeof(command), "%s comments \"%s\" --includeall --onlyfirst",
                 tb->truthbrush_path, post_id);
    }
    return run_command(command);
}

char* truthbrush_grouptags(Truthbrush* tb) {
    if (!tb) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s grouptags", tb->truthbrush_path);
    return run_command(command);
}

char* truthbrush_grouptrends(Truthbrush* tb) {
    if (!tb) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s grouptrends", tb->truthbrush_path);
    return run_command(command);
}

char* truthbrush_groupsuggestions(Truthbrush* tb) {
    if (!tb) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s groupsuggestions", tb->truthbrush_path);
    return run_command(command);
}

char* truthbrush_groupposts(Truthbrush* tb, const char* group_id) {
    if (!tb || !group_id) return NULL;
    char command[512];
    snprintf(command, sizeof(command), "%s groupposts \"%s\"",
             tb->truthbrush_path, group_id);
    return run_command(command);
}