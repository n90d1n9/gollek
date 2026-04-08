# Gollek Plugin POM Comparison

## Built-in Plugins (with Parent POM)

**Use Case**: Core plugins maintained as part of the Gollek platform

### pom.xml
```xml
<project>
    <parent>
        <groupId>tech.kayys.gollek</groupId>
        <artifactId>gollek-plugin-parent</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>
    
    <artifactId>gollek-plugin-openai</artifactId>
    
    <dependencies>
        <!-- Versions inherited from parent -->
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-spi-provider</artifactId>
            <scope>provided</scope>
        </dependency>
    </dependencies>
</project>
```

### Characteristics
- ✅ Inherits dependency management from parent
- ✅ Consistent build configuration
- ✅ Automatic plugin manifest configuration
- ✅ Auto-deploy to `~/.gollek/plugins/`
- ❌ Requires Gollek platform build
- ❌ Cannot be built standalone
- ❌ Tied to platform release cycle

### Best For
- Core platform plugins
- Plugins maintained by Gollek team
- Tight integration with platform

---

## Standalone Plugins (without Parent POM)

**Use Case**: Independent plugins by external developers

### pom.xml
```xml
<project>
    <!-- No parent - completely standalone -->
    <groupId>com.example</groupId>
    <artifactId>gollek-plugin-myprovider</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <gollek.version>1.0.0-SNAPSHOT</gollek.version>
        <maven.compiler.release>25</maven.compiler.release>
    </properties>
    
    <dependencies>
        <!-- Explicit versions -->
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-spi-provider</artifactId>
            <version>${gollek.version}</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <!-- Explicit plugin configuration -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.4.2</version>
                <configuration>
                    <archive>
                        <manifestEntries>
                            <Plugin-Id>my-provider</Plugin-Id>
                            <Plugin-Version>${project.version}</Plugin-Version>
                            <Plugin-Class>com.example.plugin.MyProvider</Plugin-Class>
                        </manifestEntries>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
    
    <profiles>
        <!-- Fat JAR for standalone deployment -->
        <profile>
            <id>fat-jar</id>
            <!-- ... shade plugin configuration -->
        </profile>
        
        <!-- Install to local plugins directory -->
        <profile>
            <id>install-plugin</id>
            <!-- ... copy to ~/.gollek/plugins/ -->
        </profile>
    </profiles>
</project>
```

### Characteristics
- ✅ No dependency on Gollek parent POM
- ✅ Can be built independently
- ✅ Independent release cycle
- ✅ Can publish to Maven Central
- ✅ External developer friendly
- ❌ Must manage all versions manually
- ❌ More boilerplate configuration
- ❌ Responsible for own dependency management

### Best For
- Third-party plugins
- Community contributions
- Commercial plugins
- Custom enterprise plugins

---

## Feature Comparison Table

| Feature | With Parent | Standalone |
|---------|-------------|------------|
| **Build Independence** | ❌ Requires platform build | ✅ Fully independent |
| **Version Management** | ✅ Inherited from parent | ⚠️ Manual configuration |
| **Dependency Management** | ✅ Centralized BOM | ⚠️ Manual or own BOM |
| **Release Cycle** | Tied to platform | Independent |
| **Maven Central Ready** | ⚠️ Requires extra config | ✅ Ready with profiles |
| **External Developer** | ❌ Not ideal | ✅ Designed for this |
| **Configuration** | ✅ Minimal boilerplate | ⚠️ More boilerplate |
| **Plugin Manifest** | ✅ Auto-configured | ⚠️ Manual configuration |
| **Deployment** | ✅ Auto-copy to plugins dir | ⚠️ Via profile or manual |
| **Fat JAR** | ⚠️ Via profile | ✅ Via profile |
| **Source/Javadoc** | ⚠️ Via profile | ✅ Via profile |
| **GPG Signing** | ⚠️ Via profile | ✅ Via profile |
| **SCM Configuration** | Platform repo | Your own repo |
| **Distribution** | Platform repos | Your choice |

---

## Migration Guide

### From Parent to Standalone

1. **Remove parent declaration**
   ```xml
   <!-- REMOVE THIS -->
   <parent>
       <groupId>tech.kayys.gollek</groupId>
       <artifactId>gollek-plugin-parent</artifactId>
       <version>1.0.0-SNAPSHOT</version>
   </parent>
   ```

2. **Add explicit coordinates**
   ```xml
   <groupId>com.example</groupId>
   <artifactId>gollek-plugin-myprovider</artifactId>
   <version>1.0.0</version>
   ```

3. **Define properties**
   ```xml
   <properties>
       <gollek.version>1.0.0-SNAPSHOT</gollek.version>
       <maven.compiler.release>25</maven.compiler.release>
   </properties>
   ```

4. **Add explicit dependency versions**
   ```xml
   <dependency>
       <groupId>tech.kayys.gollek</groupId>
       <artifactId>gollek-spi-provider</artifactId>
       <version>${gollek.version}</version>
       <scope>provided</scope>
   </dependency>
   ```

5. **Configure build plugins**
   - Copy plugin configurations from `pom-standalone.xml`

6. **Update SCM and distribution**
   - Point to your own repository
   - Configure your deployment target

### From Standalone to Parent

1. **Add parent declaration**
   ```xml
   <parent>
       <groupId>tech.kayys.gollek</groupId>
       <artifactId>gollek-plugin-parent</artifactId>
       <version>1.0.0-SNAPSHOT</version>
   </parent>
   ```

2. **Remove redundant configuration**
   - Remove explicit versions (inherited from parent)
   - Remove plugin configurations (inherited from parent)
   - Keep only plugin-specific customizations

3. **Update coordinates**
   - Use `tech.kayys.gollek` groupId
   - Follow platform naming conventions

---

## When to Use Which

### Choose Parent POM When:
- ✅ You're on the Gollek core team
- ✅ Building official platform plugins
- ✅ Want minimal configuration overhead
- ✅ Plugin is part of platform release
- ✅ Don't need independent versioning

### Choose Standalone When:
- ✅ You're an external developer
- ✅ Building third-party integrations
- ✅ Need independent release cycle
- ✅ Want to publish to Maven Central
- ✅ Building commercial/custom plugins
- ✅ Want full control over dependencies

---

## Example Scenarios

### Scenario 1: Official OpenAI Plugin
**Decision**: Use Parent POM
```
Reason: Core functionality, maintained by Gollek team, 
        released with platform
```

### Scenario 2: Company-Specific LLM Provider
**Decision**: Use Standalone
```
Reason: Internal use, custom versioning, 
        independent from platform releases
```

### Scenario 3: Community Anthropic Plugin
**Decision**: Use Standalone
```
Reason: Community contribution, 
        may be released independently, 
        developer controls release schedule
```

### Scenario 4: Experimental Provider
**Decision**: Use Standalone
```
Reason: Rapid iteration, frequent releases, 
        not ready for platform inclusion
```

---

## Hybrid Approach

You can start with standalone and migrate to parent later:

```
Development Phase (Standalone)
  ↓
Community Adoption
  ↓
Platform Integration Proposal
  ↓
Migration to Parent POM
  ↓
Official Platform Plugin
```

This allows:
- Rapid initial development
- Community validation
- Later platform integration
- No lock-in to platform cycle

---

## See Also

- `pom-standalone.xml` - Complete standalone POM template
- `PLUGIN_DEVELOPER_GUIDE.md` - Full developer guide
- `pom.xml` - Current parent-based POM example
