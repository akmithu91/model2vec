package com.model2vec;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Extracts the native library from the JAR at startup and loads it via FFM.
 * Linux x86_64 only.
 */
final class NativeLoader {

    private static volatile SymbolLookup lookup;

    private NativeLoader() {}

    static synchronized SymbolLookup load() {
        if (lookup != null) return lookup;
        final String resource = "/native/libmodel2vec_jni.so";
        try (InputStream in = NativeLoader.class.getResourceAsStream(resource)) {
            if (in == null)
                throw new UnsatisfiedLinkError(
                    "Native library not found in JAR at " + resource +
                    " — run 'mvn package' to build it.");
            Path tmp = Files.createTempFile("model2vec_native_", ".so");
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            
            // FFM library lookup
            lookup = SymbolLookup.libraryLookup(tmp.toAbsolutePath(), Arena.global());
            return lookup;
        } catch (IOException e) {
            throw new UnsatisfiedLinkError("Failed to extract native library: " + e.getMessage());
        }
    }
}
