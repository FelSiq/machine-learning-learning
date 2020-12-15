#ifndef __HASHTABLE_HT_
#define __HASHTABLE_HT_

#define STD_NEW_SIZE 10

typedef struct ht ht;

ht *htInit();
void htAll(ht *, unsigned char *, size_t);
void htPrint(ht *);
void htPurge(ht **);

#endif