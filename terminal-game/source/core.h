#ifndef __TERMINAL_CORE_H_
#define __TERMINAL_CORE_H_
#include "resources.h"
#include <stdio.h>

#define err_exit {printf("E: \"%s\" failed.\n", __FUNCTION__); return FALSE;}

enum {
	DLVL_TOP,
	DLVL_MID,
	DLVL_DEEP
};

void decodify(byte *);

#endif
