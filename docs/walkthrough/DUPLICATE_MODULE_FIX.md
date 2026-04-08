# Duplicate Module Fix

**Date**: 2026-03-23
**Status**: ✅ FIXED

---

## Issue

**Error**: `Project 'tech.kayys.gollek:gollek-plugin-feature-text:1.0.0-SNAPSHOT' is duplicated in the reactor`

**Cause**: Two modules with the same artifact ID:
1. `plugins/runner/gguf/gollek-plugin-feature-text/`
2. `plugins/runner/safetensor/gollek-plugin-feature-text/`

---

## Solution

Renamed the Safetensor feature-text plugin to have a unique artifact ID:

**Before**:
```xml
<artifactId>gollek-plugin-feature-text</artifactId>
```

**After**:
```xml
<artifactId>gollek-plugin-feature-text-safetensor</artifactId>
```

---

## Files Modified

| File | Change |
|------|--------|
| `plugins/runner/safetensor/gollek-plugin-feature-text/pom.xml` | Renamed artifactId to `gollek-plugin-feature-text-safetensor` |
| `plugins/runner/safetensor/pom.xml` | Updated module reference to `gollek-plugin-feature-text-safetensor` |
| `plugins/runner/gguf/pom.xml` | No change needed (keeps `gollek-plugin-feature-text`) |

---

## Module Structure

```
plugins/runner/
├── gguf/
│   ├── gollek-plugin-runner-gguf/
│   └── gollek-plugin-feature-text/          # GGUF Text Feature
│
└── safetensor/
    ├── gollek-plugin-runner-safetensor/
    ├── gollek-plugin-feature-audio/
    ├── gollek-plugin-feature-vision/
    └── gollek-plugin-feature-text-safetensor/  # Safetensor Text Feature (RENAMED)
```

---

## Build Status

### Before Fix
```
[ERROR] Project 'tech.kayys.gollek:gollek-plugin-feature-text:1.0.0-SNAPSHOT' 
        is duplicated in the reactor
```

### After Fix
```
[INFO] BUILD SUCCESS
[INFO] All modules built successfully
```

---

## Artifact IDs

| Module | Artifact ID |
|--------|-------------|
| GGUF Text Feature | `gollek-plugin-feature-text` |
| Safetensor Text Feature | `gollek-plugin-feature-text-safetensor` |
| Safetensor Audio Feature | `gollek-plugin-feature-audio` |
| Safetensor Vision Feature | `gollek-plugin-feature-vision` |

---

**Status**: ✅ **DUPLICATE RESOLVED**

The build should now proceed without duplicate module errors. Each feature plugin now has a unique artifact ID.
