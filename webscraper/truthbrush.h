#ifndef TRUTHBRUSH_H
#define TRUTHBRUSH_H

#include <stdlib.h>

/* Truthbrush API structure to hold configuration */
typedef struct {
    char* truthbrush_path; /* Path to the truthbrush executable */
} Truthbrush;

/* Initialize a new Truthbrush instance */
Truthbrush* truthbrush_init(const char* truthbrush_path);

/* Free a Truthbrush instance */
void truthbrush_free(Truthbrush* tb);

/* Search for users, statuses, groups, or hashtags */
char* truthbrush_search(Truthbrush* tb, const char* search_type, const char* query);

/* Pull all statuses (posts) from a user */
char* truthbrush_statuses(Truthbrush* tb, const char* handle);

/* Pull 'People to Follow' (suggested) users */
char* truthbrush_suggestions(Truthbrush* tb);

/* Pull trendy tags */
char* truthbrush_tags(Truthbrush* tb);

/* Pull ads */
char* truthbrush_ads(Truthbrush* tb);

/* Pull all of a user's metadata */
char* truthbrush_user(Truthbrush* tb, const char* handle);

/* Pull the list of users who liked a post */
char* truthbrush_likes(Truthbrush* tb, const char* post_id, int top_num);

/* Pull the list of oldest comments on a post */
char* truthbrush_comments(Truthbrush* tb, const char* post_id, int top_num);

/* Pull trending group tags */
char* truthbrush_grouptags(Truthbrush* tb);

/* Pull trending groups */
char* truthbrush_grouptrends(Truthbrush* tb);

/* Pull list of suggested groups */
char* truthbrush_groupsuggestions(Truthbrush* tb);

/* Pull posts from a group's timeline */
char* truthbrush_groupposts(Truthbrush* tb, const char* group_id);

#endif /* TRUTHBRUSH_H */