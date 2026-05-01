package tech.kayys.gollek.gguf.core;

public final class GGUFConstants {
    private GGUFConstants() {}

    public static final int GGML_TYPE_F32  = 0;
    public static final int GGML_TYPE_F16  = 1;
    public static final int GGML_TYPE_Q4_0 = 2;
    public static final int GGML_TYPE_Q4_1 = 3;
    public static final int GGML_TYPE_Q5_0 = 6;
    public static final int GGML_TYPE_Q5_1 = 7;
    public static final int GGML_TYPE_Q8_0 = 8;
    public static final int GGML_TYPE_Q2_K = 10;
    public static final int GGML_TYPE_Q3_K = 11;
    public static final int GGML_TYPE_Q4_K = 12;
    public static final int GGML_TYPE_Q5_K = 13;
    public static final int GGML_TYPE_Q6_K = 14;

    public static long tensorBytes(int type, long n) {
        return switch (type) {
            case GGML_TYPE_F32  -> n * 4;
            case GGML_TYPE_F16  -> n * 2;
            case GGML_TYPE_Q4_0 -> (n / 32) * 18;
            case GGML_TYPE_Q4_1 -> (n / 32) * 20;
            case GGML_TYPE_Q5_0 -> (n / 32) * 22;
            case GGML_TYPE_Q5_1 -> (n / 32) * 24;
            case GGML_TYPE_Q8_0 -> (n / 32) * 34;
            case GGML_TYPE_Q2_K -> (n / 256) * 84;
            case GGML_TYPE_Q3_K -> (n / 256) * 110;
            case GGML_TYPE_Q4_K -> (n / 256) * 144;
            case GGML_TYPE_Q5_K -> (n / 256) * 176;
            case GGML_TYPE_Q6_K -> (n / 256) * 210;
            default -> throw new IllegalArgumentException("Unknown type: " + type);
        };
    }
}
