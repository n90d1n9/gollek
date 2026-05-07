package tech.kayys.gollek.train;

import tech.kayys.gollek.core.tensor.Tensor;
import java.io.*;
import java.util.Map;

public final class CheckpointManager {
    public void save(String path, Map<String, Tensor> params) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(params);
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Tensor> load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            return (Map<String, Tensor>) in.readObject();
        }
    }
}