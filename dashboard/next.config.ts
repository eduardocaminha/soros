import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack(config, { isServer }) {
    if (isServer) {
      // bun:sqlite is a Bun built-in; tell webpack to leave it to the runtime
      const existing = config.externals ?? [];
      const asArray = Array.isArray(existing) ? existing : [existing];
      config.externals = [
        ...asArray,
        ({ request }: { request?: string }, callback: (err?: Error | null, result?: string) => void) => {
          if (request === "bun:sqlite") {
            return callback(null, `commonjs ${request}`);
          }
          callback();
        },
      ];
    }
    return config;
  },
};

export default nextConfig;
