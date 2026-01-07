#include <stdint.h>
__attribute__((visibility("default"))) uint32_t iJIT_GetNewMethodID(void){ static uint32_t id=0; return ++id; }
__attribute__((visibility("default"))) int iJIT_IsProfilingActive(void){ return 0; }
__attribute__((visibility("default"))) int iJIT_NotifyEvent(int eventType, void* eventData){ (void)eventType; (void)eventData; return 0; }
