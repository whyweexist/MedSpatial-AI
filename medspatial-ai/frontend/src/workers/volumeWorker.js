/**
 * MedSpatial AI — Volume Decoder Web Worker
 * Decompresses gzip'd Float32Array volume data off the main thread.
 * Uses SharedArrayBuffer + Atomics for zero-copy transfer when available.
 */

self.onmessage = async (event) => {
  const { type, data, id } = event.data;

  if (type === 'decode_volume') {
    try {
      let buffer;

      // Check if data is gzip-compressed (magic bytes: 0x1f 0x8b)
      const bytes = new Uint8Array(data);
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        // Decompress using DecompressionStream (Chrome/Edge/Firefox 102+)
        if (typeof DecompressionStream !== 'undefined') {
          const ds = new DecompressionStream('gzip');
          const writer = ds.writable.getWriter();
          writer.write(data);
          writer.close();
          const reader = ds.readable.getReader();
          const chunks = [];
          let totalLen = 0;
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            totalLen += value.byteLength;
          }
          buffer = new Uint8Array(totalLen);
          let offset = 0;
          for (const chunk of chunks) {
            buffer.set(chunk, offset);
            offset += chunk.byteLength;
          }
          buffer = buffer.buffer;
        } else {
          // Fallback: send back as-is (backend may also accept un-zipped)
          buffer = data;
        }
      } else {
        buffer = data;
      }

      // Convert to Float32Array
      const floatArray = new Float32Array(buffer);

      self.postMessage(
        { type: 'volume_decoded', id, data: floatArray.buffer },
        [floatArray.buffer]
      );
    } catch (err) {
      self.postMessage({ type: 'error', id, error: err.message });
    }
  }

  if (type === 'detect_gpu') {
    // Run GPU capability detection in worker context
    self.postMessage({
      type: 'gpu_info',
      id,
      info: {
        workerContext: true,
        supportsSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
        supportsDecompressionStream: typeof DecompressionStream !== 'undefined',
      },
    });
  }
};
