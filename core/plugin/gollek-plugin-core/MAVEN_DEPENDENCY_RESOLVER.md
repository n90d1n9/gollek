# Plugin System with Maven Dependency Resolution

## 🎯 Overview

The Gollek Plugin System now supports **automatic Maven dependency resolution** for plugins. When a plugin is loaded, its dependencies declared in `plugin.json` are automatically resolved from Maven repositories and added to the plugin's ClassLoader.

---

## 📦 Architecture

```
Plugin JAR Loading Flow:

1. Discover JAR in ~/.gollek/plugins/
         ↓
2. Read plugin.json descriptor
         ↓
3. Parse "dependencies" array
         ↓
4. Resolve Maven dependencies
   - Check local cache (~/.m2/repository)
   - Download from remote repos if needed
   - Resolve transitive dependencies
         ↓
5. Create ClassLoader with:
   - Plugin JAR
   - All dependency JARs
         ↓
6. Load and instantiate plugin class
```

---

## 🔧 Plugin Descriptor (plugin.json)

### Dependencies Section

```json
{
  "id": "openai-cloud-provider",
  "name": "OpenAI Cloud Provider",
  "version": "1.0.0",
  "mainClass": "tech.kayys.gollek.plugin.cloud.openai.OpenAiCloudProvider",
  
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT",
    "com.squareup.okhttp3:okhttp:4.12.0",
    "com.fasterxml.jackson.core:jackson-databind:2.16.0"
  ],
  
  "capabilities": ["cloud-provider", "streaming"]
}
```

### Dependency Format

Dependencies use Maven coordinate format:

```
groupId:artifactId:version
```

**Examples:**
- `tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT`
- `com.squareup.okhttp3:okhttp:4.12.0`
- `org.apache.commons:commons-lang3:3.14.0`

**Version Ranges:**
- `[1.0,2.0)` - Version range 1.0 (inclusive) to 2.0 (exclusive)
- `[1.0,)` - Version 1.0 or higher
- `LATEST` - Latest release
- `RELEASE` - Latest release (non-snapshot)

---

## 📥 Dependency Resolution

### Resolution Order

1. **Local Repository** (`~/.m2/repository`)
   - Check if artifact exists
   - Use cached version if available

2. **Remote Repositories** (if not in local)
   - Maven Central (https://repo.maven.apache.org/maven2)
   - JBoss Public (https://repository.jboss.org/nexus/content/groups/public/)
   - Custom repositories (configured in settings)

3. **Transitive Dependencies**
   - Automatically resolved
   - Version conflicts resolved using "nearest definition" strategy

### Configuration

#### Default Repositories

```java
MavenDependencyResolver resolver = new MavenDependencyResolver();
// Uses:
// - Local: ~/.m2/repository
// - Remote: Maven Central, JBoss Public
```

#### Custom Local Repository

```java
MavenDependencyResolver resolver = new MavenDependencyResolver(
    "/path/to/custom/m2/repository"
);
```

#### Add Custom Remote Repository

```java
MavenDependencyResolver resolver = new MavenDependencyResolver();

// Add public repository
resolver.addRepository("my-repo", "https://repo.example.com/maven2", null, null);

// Add private repository with authentication
resolver.addRepository(
    "private-repo", 
    "https://repo.company.com/maven2",
    "username", 
    "password"
);
```

#### System Properties

```bash
# Custom local repository
java -Dmaven.repo.local=/path/to/repo -jar gollek-runtime.jar

# Custom plugin directory
java -Dgollek.plugin.directory=/path/to/plugins -jar gollek-runtime.jar
```

---

## 🚀 Usage Examples

### Example 1: Simple Plugin with Dependencies

**plugin.json:**
```json
{
  "id": "my-plugin",
  "mainClass": "com.example.MyPlugin",
  "dependencies": [
    "org.apache.commons:commons-lang3:3.14.0"
  ]
}
```

**Result:**
- `commons-lang3-3.14.0.jar` automatically downloaded
- Added to plugin's ClassLoader
- Plugin can use `org.apache.commons.lang3.*` classes

### Example 2: Plugin with Transitive Dependencies

**plugin.json:**
```json
{
  "id": "http-client-plugin",
  "mainClass": "com.example.HttpClientPlugin",
  "dependencies": [
    "com.squareup.okhttp3:okhttp:4.12.0"
  ]
}
```

**Result:**
- OkHttp 4.12.0 downloaded
- **Transitive dependencies also resolved:**
  - `okio:okio:3.6.0`
  - `org.jetbrains.kotlin:kotlin-stdlib:1.9.20`
- All JARs added to ClassLoader

### Example 3: Plugin with Multiple Dependencies

**plugin.json:**
```json
{
  "id": "advanced-plugin",
  "mainClass": "com.example.AdvancedPlugin",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT",
    "com.fasterxml.jackson.core:jackson-databind:2.16.0",
    "org.apache.httpcomponents.client5:httpclient5:5.3"
  ]
}
```

**Result:**
- All 4 direct dependencies resolved
- All transitive dependencies resolved (~20+ JARs total)
- Single ClassLoader with all JARs

---

## 📊 Dependency Resolution API

### Manual Resolution

```java
import tech.kayys.gollek.plugin.core.MavenDependencyResolver;

// Create resolver
MavenDependencyResolver resolver = new MavenDependencyResolver();

// Resolve single dependency
List<File> jars = resolver.resolve("com.squareup.okhttp3:okhttp:4.12.0");

// Resolve multiple dependencies
List<String> deps = List.of(
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
);
List<File> allJars = resolver.resolveAll(deps);

// Get classpath string
String classpath = resolver.resolveClasspath("com.example:lib:1.0.0");
// Returns: "/home/user/.m2/repository/com/example/lib/1.0.0/lib-1.0.0.jar"
```

### Build Dependency Tree

```java
// Build dependency tree
List<String> deps = List.of("com.squareup.okhttp3:okhttp:4.12.0");
MavenDependencyResolver.DependencyTree tree = resolver.buildTree(deps);

// Print tree
System.out.println(tree);
```

**Output:**
```
+- com.squareup.okhttp3:okhttp:4.12.0
  +- com.squareup.okio:okio:3.6.0
    +- com.squareup.okio:okio-jvm:3.6.0
      +- org.jetbrains.kotlin:kotlin-stdlib:1.9.20
        +- org.jetbrains.kotlin:kotlin-stdlib-common:1.9.20
        +- org.jetbrains:annotations:13.0
```

---

## 🔍 Troubleshooting

### Dependency Not Found

**Error:**
```
Failed to resolve dependency: com.example:lib:1.0.0
```

**Solutions:**

1. **Check coordinate format**
   ```json
   // Correct: groupId:artifactId:version
   "com.example:example-lib:1.0.0"
   
   // Wrong: missing version
   "com.example:example-lib"
   ```

2. **Verify artifact exists in repository**
   ```bash
   # Check Maven Central
   curl https://repo.maven.apache.org/maven2/com/example/example-lib/1.0.0/
   ```

3. **Add custom repository**
   ```java
   resolver.addRepository("my-repo", "https://repo.example.com/maven2", null, null);
   ```

### Version Conflict

**Error:**
```
Version conflict: com.google.guava:guava:30.0 vs 32.0
```

**Solution:**
- Use explicit versions in `plugin.json`
- Use dependency management in parent POM
- Exclude transitive dependencies if needed

### Snapshot Not Updating

**Issue:** Plugin uses SNAPSHOT version but old version cached

**Solution:**
```java
// Clear cache
resolver.clearCache();

// Or force update
System.setProperty("maven.updatePolicy", "always");
```

### Authentication Required

**Error:**
```
Failed to resolve: authentication required
```

**Solution:**
```java
// Add repository with credentials
resolver.addRepository(
    "private-repo",
    "https://repo.company.com/maven2",
    "username",
    "password"
);
```

Or use `~/.m2/settings.xml`:
```xml
<settings>
  <servers>
    <server>
      <id>private-repo</id>
      <username>myuser</username>
      <password>mypassword</password>
    </server>
  </servers>
</settings>
```

---

## 📁 File Locations

### Plugin JARs
```
~/.gollek/plugins/
├── openai-provider.jar
├── anthropic-provider.jar
└── plugin.json
```

### Maven Dependencies
```
~/.m2/repository/
├── tech/kayys/
│   ├── gollek-spi-provider/
│   │   └── 1.0.0-SNAPSHOT/
│   │       ├── gollek-spi-provider-1.0.0-SNAPSHOT.jar
│   │       └── maven-metadata.xml
│   └── gollek-spi-inference/
│       └── 1.0.0-SNAPSHOT/
│           └── gollek-spi-inference-1.0.0-SNAPSHOT.jar
├── com/squareup/okhttp3/
│   └── okhttp/
│       └── 4.12.0/
│           └── okhttp-4.12.0.jar
└── ...
```

### Plugin ClassLoader
```
Plugin ClassLoader
├── file:~/.gollek/plugins/openai-provider.jar
├── file:~/.m2/repository/tech/kayys/gollek-spi-provider/1.0.0-SNAPSHOT/gollek-spi-provider-1.0.0-SNAPSHOT.jar
├── file:~/.m2/repository/tech/kayys/gollek-spi-inference/1.0.0-SNAPSHOT/gollek-spi-inference-1.0.0-SNAPSHOT.jar
├── file:~/.m2/repository/com/squareup/okhttp3/okhttp/4.12.0/okhttp-4.12.0.jar
└── ... (all transitive dependencies)
```

---

## 🎯 Best Practices

### 1. Use Explicit Versions

```json
// ✅ Good - explicit version
"dependencies": [
  "com.squareup.okhttp3:okhttp:4.12.0"
]

// ❌ Avoid - version ranges can be unpredictable
"dependencies": [
  "com.squareup.okhttp3:okhttp:[4.0,)"
]
```

### 2. Minimize Dependencies

```json
// ✅ Good - only what's needed
"dependencies": [
  "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT"
]

// ❌ Avoid - too many dependencies
"dependencies": [
  "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
  "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT",
  "tech.kayys:gollek-spi-model:1.0.0-SNAPSHOT",
  "tech.kayys:gollek-spi-plugin:1.0.0-SNAPSHOT",
  "com.fasterxml.jackson.core:jackson-databind:2.16.0",
  "org.apache.commons:commons-lang3:3.14.0",
  "com.google.guava:guava:32.0.0-jre",
  "org.apache.httpcomponents:httpclient:4.5.14"
]
```

### 3. Use Provided Scope for Gollek APIs

Gollek SPI modules are already available at runtime:

```json
// ✅ Good - minimal dependencies
"dependencies": [
  "com.squareup.okhttp3:okhttp:4.12.0"
]

// ❌ Avoid - Gollek SPIs already provided
"dependencies": [
  "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",  // Already provided
  "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT", // Already provided
  "com.squareup.okhttp3:okhttp:4.12.0"
]
```

### 4. Test Dependencies Locally First

```bash
# Test plugin with mvn
cd my-plugin
mvn clean package

# Copy to plugins directory
cp target/my-plugin-1.0.0.jar ~/.gollek/plugins/

# Check logs for dependency resolution
tail -f ~/.gollek/logs/gollek.log | grep -i dependency
```

### 5. Monitor Dependency Size

```bash
# Check total size of resolved dependencies
du -sh ~/.m2/repository/tech/kayys/
du -sh ~/.m2/repository/com/squareup/okhttp3/

# Large dependencies may slow plugin loading
```

---

## 📊 Performance

### Resolution Time

| Scenario | Time | Notes |
|----------|------|-------|
| Cached (local repo) | <100ms | From ~/.m2/repository |
| First download | 1-5s | Depends on size & network |
| Transitive resolution | +500ms | Per level of depth |
| Large dependency tree | 5-10s | 50+ JARs |

### Optimization Tips

1. **Pre-download dependencies**
   ```bash
   # Download before deployment
   mvn dependency:copy-dependencies -DoutputDirectory=~/.m2/repository
   ```

2. **Use fat JAR for complex plugins**
   ```bash
   mvn clean package -Pfat-jar
   # Includes all dependencies in single JAR
   ```

3. **Cache warm-up**
   ```java
   // Pre-resolve common dependencies at startup
   resolver.resolve("tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT");
   ```

---

## 🔗 Resources

- [Maven Dependency Mechanism](https://maven.apache.org/guides/introduction/introduction-to-dependency-mechanism.html)
- [Maven Repository Search](https://search.maven.org/)
- [Eclipse Aether Documentation](https://eclipse.org/aether/)
- [Plugin System Guide](/docs/plugin-system)

---

**Version**: 2.1.0  
**Last Updated**: 2026-03-22  
**Status**: ✅ Production Ready
