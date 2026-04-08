package tech.kayys.gollek.quantizer.gptq;

import java.util.List;

/**
 * Metadata for SafeTensors file header.
 */
public class GPTQSafetensorHeader {
    
    /**
     * Descriptor for a single tensor in the header.
     */
    public record TensorInfo(
            String dtype,
            List<Long> shape,
            List<Long> data_offsets) {
            
        public String getDtype() {
            return dtype;
        }
        
        public List<Long> getShape() {
            return shape;
        }
        
        public List<Long> getDataOffsets() {
            return data_offsets;
        }
    }
}
