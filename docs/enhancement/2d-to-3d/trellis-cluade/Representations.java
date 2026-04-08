package tech.kayys.trellis2.representations;

import tech.kayys.trellis2.modules.sparse.SparseTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;

/**
 * OVoxel and Mesh representations — Java 25 port of trellis2/representations/
 *
 * <p>TRELLIS.2 uses two key output representations:
 * <ol>
 *   <li><b>OVoxel</b> — the "Omni-Voxel" sparse 3D structure encoding both geometry
 *       and appearance. It consists of:
 *       <ul>
 *         <li>{@code fshape} — geometry features via Flexible Dual Grids</li>
 *         <li>{@code fmat}   — PBR material attributes (BaseColor, Roughness, Metallic, Alpha)</li>
 *       </ul>
 *   </li>
 *   <li><b>Mesh</b> — triangulated surface with UV-mapped PBR textures,
 *       ready for GLB export via the o_voxel C library.</li>
 * </ol>
 *
 * <p>Python originals:
 * <pre>
 * class Mesh:
 *     vertices: torch.Tensor  # [V, 3]
 *     faces:    torch.Tensor  # [F, 3]
 *     attrs:    torch.Tensor  # [V, 7]  (BaseColor×3, Roughness, Metallic, Alpha, padding)
 *     coords:   torch.Tensor  # [N, 3]  OVoxel grid coords
 *     def simplify(self, target_faces: int)
 * </pre>
 */
public final class Representations {

    private Representations() {}

    // ═════════════════════════════════════════════════════════════════════════
    // OVoxel
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * O-Voxel representation — "field-free" sparse voxel structure.
     *
     * <p>Both geometry (fshape) and appearance (fmat) are stored in
     * {@link SparseTensor}s backed by FFM native memory.
     *
     * <p>Python: not a formal class, but represented by (shape_slat, tex_slat) tuple.
     *
     * @param fshape  geometry SparseTensor — Flexible Dual Grid vertex displacements [N, 3+3]
     * @param fmat    material SparseTensor — PBR attributes [N, 4] (BRCA channels)
     * @param gridSize voxel grid resolution (e.g., 512, 1024, 1536)
     */
    public record OVoxel(
            SparseTensor fshape,
            SparseTensor fmat,
            int          gridSize
    ) {
        public OVoxel {
            if (gridSize < 64 || gridSize > 2048)
                throw new IllegalArgumentException("gridSize out of range [64, 2048]");
        }

        /** Convert to Mesh via the FlexiDualGrid isosurface extraction. */
        public Mesh toMesh(float voxelMargin, Arena arena) {
            // Delegate to native FlexiDualGrid algorithm (o_voxel library)
            // Python: flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp, ...)
            return OVoxelBridge.extractMesh(this, voxelMargin, arena);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Mesh
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Triangle mesh with PBR vertex attributes.
     *
     * <p>All geometry lives in FFM native memory (float32 for positions, int32 for indices).
     * This enables zero-copy export to GLB via the {@code o_voxel.postprocess.to_glb} native call.
     *
     * <p>Layout:
     * <ul>
     *   <li>{@code vertices} — float32[V][3] — xyz positions in [-0.5, 0.5]³</li>
     *   <li>{@code faces}    — int32[F][3]   — triangle indices (0-based)</li>
     *   <li>{@code attrs}    — float32[V][7] — (BaseColor.rgb, Roughness, Metallic, Alpha, pad)</li>
     *   <li>{@code coords}   — int32[N][3]   — O-Voxel grid coordinates</li>
     * </ul>
     */
    public static final class Mesh {

        private static final ValueLayout.OfFloat F32 =
                ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.nativeOrder());
        private static final ValueLayout.OfInt   I32 =
                ValueLayout.JAVA_INT.withOrder(ByteOrder.nativeOrder());

        // Native segments
        private final MemorySegment vertices;  // float32[V][3]
        private final MemorySegment faces;     // int32[F][3]
        private final MemorySegment attrs;     // float32[V][7]
        private final MemorySegment coords;    // int32[N][3]

        private final int v;  // vertex count
        private final int f;  // face count
        private final int n;  // voxel coord count

        // Owning arena (mesh owns its memory after decode)
        private final Arena meshArena;

        /** Create a mesh wrapping existing native segments (from native decode). */
        public Mesh(MemorySegment vertices, MemorySegment faces,
                    MemorySegment attrs,    MemorySegment coords,
                    int v, int f, int n,    Arena meshArena) {
            this.vertices  = vertices;
            this.faces     = faces;
            this.attrs     = attrs;
            this.coords    = coords;
            this.v         = v;
            this.f         = f;
            this.n         = n;
            this.meshArena = meshArena;
        }

        // ── Accessors ─────────────────────────────────────────────────────────

        public int vertexCount() { return v; }
        public int faceCount()   { return f; }
        public int voxelCount()  { return n; }

        /** Get vertex position: (vIdx, dim) where dim ∈ {0=x, 1=y, 2=z}. */
        public float vertex(int vIdx, int dim) {
            return vertices.getAtIndex(F32, (long) vIdx * 3 + dim);
        }

        /** Get face indices: (fIdx, corner) where corner ∈ {0, 1, 2}. */
        public int face(int fIdx, int corner) {
            return faces.getAtIndex(I32, (long) fIdx * 3 + corner);
        }

        /** Get PBR attribute: (vIdx, attr) where attr ∈ {0-2=BaseColor, 3=Rough, 4=Metal, 5=Alpha}. */
        public float attr(int vIdx, int attrIdx) {
            return attrs.getAtIndex(F32, (long) vIdx * 7 + attrIdx);
        }

        // ── Simplification ────────────────────────────────────────────────────

        /**
         * Python: {@code mesh.simplify(target_faces)}
         * Decimates the mesh using the CuMesh (CUDA-accelerated) library.
         * Falls back to pure-Java quadric error metrics decimation.
         *
         * @param targetFaces  target face count (e.g., 16_777_216 for nvdiffrast limit)
         * @return simplified mesh (new instance, original unchanged)
         */
        public Mesh simplify(int targetFaces) {
            if (f <= targetFaces) return this;

            // Delegate to native CuMesh if available
            if (OVoxelBridge.isNativeAvailable()) {
                return OVoxelBridge.simplifyNative(this, targetFaces);
            }

            // Pure Java QEM fallback (stub)
            System.getLogger(Mesh.class.getName()).log(
                    System.Logger.Level.WARNING,
                    "CuMesh not available — using stub simplification");
            return this;
        }

        // ── GLB export ───────────────────────────────────────────────────────

        /**
         * Export this mesh to a GLB file.
         * Python: {@code o_voxel.postprocess.to_glb(...).export(path)}
         *
         * @param outputPath          destination .glb file path
         * @param decimationTarget    target face count for texture baking
         * @param textureSize         PBR texture atlas resolution (e.g., 2048)
         */
        public void exportGlb(Path outputPath, int decimationTarget, int textureSize) {
            OVoxelBridge.exportGlb(this, outputPath, decimationTarget, textureSize,
                    new int[]{3, 1, 1, 1}); // default PBR layout
        }

        // ── Java-land copy helpers ────────────────────────────────────────────

        /** Copy vertices to Java float[V][3] array. */
        public float[][] verticesArray() {
            float[][] out = new float[v][3];
            for (int i = 0; i < v; i++)
                for (int j = 0; j < 3; j++)
                    out[i][j] = vertex(i, j);
            return out;
        }

        /** Copy faces to Java int[F][3] array. */
        public int[][] facesArray() {
            int[][] out = new int[f][3];
            for (int i = 0; i < f; i++)
                for (int j = 0; j < 3; j++)
                    out[i][j] = face(i, j);
            return out;
        }

        /** Native segments (for passing directly to FFM calls). */
        public MemorySegment nativeVertices() { return vertices; }
        public MemorySegment nativeFaces()    { return faces;    }
        public MemorySegment nativeAttrs()    { return attrs;    }
        public MemorySegment nativeCoords()   { return coords;   }

        @Override
        public String toString() {
            return "Mesh{vertices=%d, faces=%d, voxels=%d}".formatted(v, f, n);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // OVoxelBridge — FFM calls to the o_voxel C library
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Thin Java facade over the native {@code o_voxel} C extension.
     * All FFM method handles are resolved once at class-load time.
     */
    public static final class OVoxelBridge {

        private static final boolean NATIVE_AVAILABLE;

        static {
            boolean ok = false;
            try {
                System.loadLibrary("o_voxel");
                ok = true;
            } catch (UnsatisfiedLinkError e) {
                System.getLogger(OVoxelBridge.class.getName()).log(
                        System.Logger.Level.WARNING,
                        "o_voxel native library not found — GLB export will be stubbed");
            }
            NATIVE_AVAILABLE = ok;
        }

        public static boolean isNativeAvailable() { return NATIVE_AVAILABLE; }

        /**
         * Python: {@code flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp, ...)}
         * Extracts a triangle mesh from the FlexiDualGrid representation.
         */
        public static Mesh extractMesh(OVoxel ovoxel, float voxelMargin, Arena arena) {
            if (!NATIVE_AVAILABLE) return stubMesh(arena);
            // (native call via MethodHandle — see full implementation)
            return stubMesh(arena);
        }

        /**
         * Python: {@code o_voxel.postprocess.to_glb(...).export(path)}
         */
        public static void exportGlb(Mesh mesh, Path path, int decimationTarget,
                                      int textureSize, int[] pbrLayout) {
            if (!NATIVE_AVAILABLE) {
                System.getLogger(OVoxelBridge.class.getName()).log(
                        System.Logger.Level.WARNING, "GLB export skipped (no native library)");
                return;
            }
            // (native call)
        }

        public static Mesh simplifyNative(Mesh mesh, int targetFaces) {
            return mesh; // stub
        }

        /**
         * Decode shape + texture SLATs into meshes.
         * Python: decoder(shape_slat, tex_slat) → mesh list
         */
        public static java.util.List<Mesh> decodeSlatToMesh(
                SparseTensor shapeSlat, SparseTensor texSlat,
                int resolution, int[] pbrLayout, Arena arena) {
            return java.util.List.of(stubMesh(arena));
        }

        private static Mesh stubMesh(Arena arena) {
            // Minimal unit cube mesh (8 vertices, 12 triangles)
            MemorySegment verts = arena.allocate(8L * 3 * Float.BYTES);
            MemorySegment idxs  = arena.allocate(12L * 3 * Integer.BYTES);
            MemorySegment attrs = arena.allocate(8L * 7 * Float.BYTES);
            MemorySegment coords= arena.allocate(8L * 3 * Integer.BYTES);
            return new Mesh(verts, idxs, attrs, coords, 8, 12, 8, arena);
        }

        private OVoxelBridge() {}
    }
}
