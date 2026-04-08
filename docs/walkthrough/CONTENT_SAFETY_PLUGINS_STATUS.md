# Content Safety Plugins - Status and Recommendations

**Date**: 2026-03-23
**Status**: ⚠️ **TEMPORARILY EXCLUDED - NEEDS REFACTORING**

---

## Plugins in Question

### 1. ContentSafetyPlugin
**Location**: `plugins/common/gollek-plugin-content-safety/`

**Purpose**: Content moderation and safety validation

**Features**:
- Keyword blocking
- Regex pattern matching
- Content filtering before inference
- Phase-bound to VALIDATE phase

### 2. DefaultContentModerator
**Purpose**: Default content moderation implementation

### 3. RequestValidationPlugin
**Purpose**: Request validation before processing

---

## Are These Plugins Meaningful/Usable?

### ✅ **YES - Highly Valuable for Production**

These plugins provide **critical production functionality**:

#### Content Safety (DefaultContentModerator, ContentSafetyPlugin)
**Use Cases**:
- ✅ Filter harmful/inappropriate content
- ✅ Block hate speech, violence, adult content
- ✅ Compliance with content policies
- ✅ Protect users from harmful outputs
- ✅ Regulatory compliance (GDPR, COPPA, etc.)

**Production Value**: **HIGH**
- Essential for public-facing AI services
- Required for enterprise deployments
- Critical for compliance and liability protection

#### Request Validation (RequestValidationPlugin)
**Use Cases**:
- ✅ Validate input before processing
- ✅ Prevent injection attacks
- ✅ Enforce input length limits
- ✅ Validate request format
- ✅ Rate limiting support

**Production Value**: **HIGH**
- Security hardening
- Input sanitization
- Attack prevention

---

## Why Temporarily Excluded?

### Technical Issue

**Problem**: These plugins implement `InferencePhasePlugin` interface which doesn't exist in the current SPI.

```java
public class ContentSafetyPlugin implements InferencePhasePlugin {
    // InferencePhasePlugin doesn't exist in current SPI
}
```

**Root Cause**: The `InferencePhasePlugin` interface was part of an older architecture that's not in the current plugin SPI.

### Options to Fix

#### Option 1: Create InferencePhasePlugin Interface (Recommended)
**Effort**: 2-4 hours

**Steps**:
1. Create `InferencePhasePlugin` interface in `gollek-spi-plugin`
2. Define phase-bound execution model
3. Update content safety plugins to use new interface
4. Add phase execution support to plugin manager

**Benefits**:
- Proper phase-bound plugin architecture
- Clean separation of concerns
- Extensible for other phase plugins

#### Option 2: Refactor to Simple Plugins
**Effort**: 4-6 hours

**Steps**:
1. Remove `InferencePhasePlugin` dependency
2. Implement as simple `GollekPlugin` implementations
3. Add content safety checks to inference flow manually
4. Update plugin manager to call content safety

**Benefits**:
- Simpler architecture
- No phase system needed

**Drawbacks**:
- Less flexible
- Manual integration required

#### Option 3: Exclude Permanently (Not Recommended)
**Effort**: 0 hours

**Impact**:
- Lose critical production functionality
- Need to implement content safety manually in each application
- Compliance risks

---

## Recommendation

### ✅ **FIX AND INCLUDE THESE PLUGINS**

**Reason**: Content safety and request validation are **critical for production deployments**.

**Recommended Approach**: **Option 1** - Create `InferencePhasePlugin` interface

**Estimated Effort**: 2-4 hours

**Implementation Plan**:

1. **Create InferencePhasePlugin SPI** (1 hour)
   ```java
   public interface InferencePhasePlugin extends GollekPlugin {
       InferencePhase phase();
       int order();
       boolean shouldExecute(ExecutionContext context);
       void execute(ExecutionContext context, EngineContext engine);
   }
   ```

2. **Create InferencePhase enum** (30 minutes)
   ```java
   public enum InferencePhase {
       VALIDATE, PRE_PROCESS, INFERENCE, POST_PROCESS
   }
   ```

3. **Update content safety plugins** (1 hour)
   - Fix imports
   - Implement interface methods
   - Test compilation

4. **Add phase execution to plugin manager** (1 hour)
   - Phase-aware plugin execution
   - Order-based execution within phases
   - Test integration

**Total Effort**: ~3.5 hours

---

## Current Status

| Plugin | Status | Production Ready | Notes |
|--------|--------|-----------------|-------|
| ContentSafetyPlugin | ⚠️ Excluded | NO | Needs InferencePhasePlugin |
| DefaultContentModerator | ⚠️ Excluded | NO | Needs InferencePhasePlugin |
| RequestValidationPlugin | ⚠️ Excluded | NO | Needs InferencePhasePlugin |

---

## Alternative (Temporary Workaround)

If you need content safety **immediately** without fixing the plugins:

### Manual Integration

```java
// In your inference service
public class InferenceService {
    
    @Inject
    ContentSafetyPlugin contentSafety;
    
    public InferenceResponse infer(InferenceRequest request) {
        // Manual content safety check
        SafetyValidationResult result = contentSafety.validate(request);
        if (!result.isSafe()) {
            throw new ContentSafetyException("Content unsafe: " + result.violations());
        }
        
        // Proceed with inference
        return runner.infer(request);
    }
}
```

**Drawbacks**:
- Manual integration in each service
- Not centralized
- Harder to maintain
- No phase-based execution

---

## Conclusion

**Are these plugins meaningful/usable?**

✅ **YES - Absolutely!**

These plugins provide **critical production functionality**:
- Content safety and moderation
- Request validation
- Security hardening
- Compliance support

**Recommendation**: Fix the `InferencePhasePlugin` dependency (3-4 hours) and include these plugins in the plugin system.

**Temporary Status**: Excluded from build until fixed.

**Priority**: **HIGH** - Should be fixed before production deployment.

---

## Next Steps

1. ✅ Acknowledge these plugins are valuable
2. ⏳ Decide on fix approach (Option 1 recommended)
3. ⏳ Allocate 3-4 hours for fix
4. ⏳ Implement InferencePhasePlugin SPI
5. ⏳ Update content safety plugins
6. ⏳ Test and include in plugin system

---

**Status**: ⚠️ **TEMPORARILY EXCLUDED - HIGH PRIORITY TO FIX**
