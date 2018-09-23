///////////////////////////////////////////////////////////////////////////
//
// NAME
//  Error.h -- a simple error handling class
//
// DESCRIPTION
//  The CError class is used to throw error messages back to the calling program.
//
// Copyright © Richard Szeliski, 2001.
// See Copyright.h for more details
// modified to prevent buffer overflow DS 3/28/2014
//
///////////////////////////////////////////////////////////////////////////

namespace std {}
using namespace std;

#include <string.h>
#include <stdio.h>
#include <exception>
#define MSGLEN 10000 // longest allowable message

struct CError : public exception
{
    CError(const char* msg)                       { snprintf(message, MSGLEN, "%s", msg); }
    CError(const char* fmt, int d)                { snprintf(message, MSGLEN, fmt, d); }
    CError(const char* fmt, float f)              { snprintf(message, MSGLEN, fmt, f); }
    CError(const char* fmt, const char *s)        { snprintf(message, MSGLEN, fmt, s); }
    CError(const char* fmt, const char *s, int d) { snprintf(message, MSGLEN, fmt, s, d); }
    char message[MSGLEN];
};
