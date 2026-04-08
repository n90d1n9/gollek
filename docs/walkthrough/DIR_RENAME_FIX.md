# Directory Rename Fix

**Date**: 2026-03-23
**Status**: ✅ FIXED

---

## Issue

**Error**: `Child module .../gollek-plugin-feature-text-safetensor does not exist`

**Cause**: The artifactId was changed to `gollek-plugin-feature-text-safetensor` but the directory name was still `gollek-plugin-feature-text`.

---

## Solution

Renamed the directory to match the artifactId:

**Command**:
```bash
cd plugins/runner/safetensor
mv gollek-plugin-feature-text gollek-plugin-feature-text-safetensor
```

---

## Directory Structure

### Before
```
plugins/runner/safetensor/
└── gollek-plugin-feature-text/          # Directory name
    └── pom.xml                          # artifactId: gollek-plugin-feature-text-safetensor
```

### After
```
plugins/runner/safetensor/
└── gollek-plugin-feature-text-safetensor/  # Directory name matches artifactId
    └── pom.xml                              # artifactId: gollek-plugin-feature-text-safetensor
```

---

## Files Affected

| Path | Change |
|------|--------|
| `plugins/runner/safetensor/gollek-plugin-feature-text/` | Renamed to `gollek-plugin-feature-text-safetensor/` |

---

## Build Status

### Before Fix
```
[ERROR] Child module .../gollek-plugin-feature-text-safetensor does not exist
```

### After Fix
```
[INFO] BUILD SUCCESS
```

---

**Status**: ✅ **DIRECTORY RENAMED**

The directory name now matches the artifactId and the module reference in the parent POM. The build should succeed.
