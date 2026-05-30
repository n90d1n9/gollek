package tech.kayys.gollek.jupyter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class NotebookMagicArgs {

    private NotebookMagicArgs() {
    }

    record OptionSpec(String name, boolean requiresValue, List<String> aliases) {
        boolean matches(String token) {
            return aliases.contains(token);
        }
    }

    record ParsedOptions(Map<String, String> values, Set<String> flags, List<String> optionOrder, List<String> positionals) {
        boolean has(String name) {
            return flags.contains(name) || values.containsKey(name);
        }

        String value(String name) {
            return values.get(name);
        }

        String lastPresent(String... names) {
            Set<String> wanted = Set.copyOf(Arrays.asList(names));
            for (int i = optionOrder.size() - 1; i >= 0; i--) {
                String name = optionOrder.get(i);
                if (wanted.contains(name)) {
                    return name;
                }
            }
            return null;
        }

        List<String> requirePositionals(int count, String usage) {
            if (positionals.size() != count || positionals.stream().anyMatch(String::isBlank)) {
                throw new IllegalArgumentException(usage);
            }
            return positionals;
        }

        List<String> requirePositionalsBetween(int min, int max, String usage) {
            if (positionals.size() < min || positionals.size() > max || positionals.stream().anyMatch(String::isBlank)) {
                throw new IllegalArgumentException(usage);
            }
            return positionals;
        }
    }

    static OptionSpec flag(String name, String... aliases) {
        return new OptionSpec(name, false, List.copyOf(Arrays.asList(aliases)));
    }

    static OptionSpec value(String name, String... aliases) {
        return new OptionSpec(name, true, List.copyOf(Arrays.asList(aliases)));
    }

    static ParsedOptions parseLeadingOptions(String raw, String usage, String unknownPrefix, OptionSpec... specs) {
        String trimmed = raw == null ? "" : raw.trim();
        if (trimmed.isBlank()) {
            return new ParsedOptions(Map.of(), Set.of(), List.of(), List.of());
        }
        String[] tokens = trimmed.split("\\s+");
        Map<String, String> values = new LinkedHashMap<>();
        Set<String> flags = new LinkedHashSet<>();
        List<String> optionOrder = new ArrayList<>();
        int index = 0;
        while (index < tokens.length) {
            String token = tokens[index];
            OptionSpec spec = findSpec(token, specs);
            if (spec == null) {
                if (!token.startsWith("--")) {
                    break;
                }
                throw new IllegalArgumentException(unknownPrefix + token);
            }
            optionOrder.add(spec.name());
            if (spec.requiresValue()) {
                if (index + 1 >= tokens.length) {
                    throw new IllegalArgumentException(usage);
                }
                values.put(spec.name(), tokens[index + 1]);
                index += 2;
            } else {
                flags.add(spec.name());
                index++;
            }
        }
        List<String> positionals = new ArrayList<>();
        for (int i = index; i < tokens.length; i++) {
            positionals.add(tokens[i]);
        }
        return new ParsedOptions(Map.copyOf(values), Set.copyOf(flags), List.copyOf(optionOrder), List.copyOf(positionals));
    }

    private static OptionSpec findSpec(String token, OptionSpec[] specs) {
        for (OptionSpec spec : specs) {
            if (spec.matches(token)) {
                return spec;
            }
        }
        return null;
    }
}
