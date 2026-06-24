import tech.kayys.aljabr.metal.binding.MetalBinding;

public class TestMetal {
    public static void main(String[] args) {
        boolean initialized = MetalBinding.initialize();
        System.out.println("Initialized: " + initialized);
        MetalBinding binding = MetalBinding.getInstance();
        binding.init();
        System.out.println("Active: " + binding.isRuntimeActive());
        System.out.println("Device: " + binding.deviceName());
    }
}
