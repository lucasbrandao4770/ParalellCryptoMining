#ifndef SHA256_CUH
#define SHA256_CUH

#include <stdint.h>

/****************************** MACROS ******************************/
/// Length of the SHA-256 digest in bytes (32 bytes)
#define SHA256_DIGEST_LENGTH 32

/// Block size for SHA-256, which is also 32 bytes
#define SHA256_BLOCK_SIZE 32

/// Left rotation of 32-bit word
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

/// Right rotation of 32-bit word
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

/// SHA-256 Choose function
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))

/// SHA-256 Majority function
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

/// SHA-256 Sigma 0 function
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))

/// SHA-256 Sigma 1 function
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))

/// SHA-256 Small Sigma 0 function
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))

/// SHA-256 Small Sigma 1 function
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/// Macro for checking and reporting CUDA errors
#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
        printf("GPU: cudaError %d (%s)\n", err, cudaGetErrorString(err)); \
}


/**************************** DATA TYPES ****************************/
/// 8-bit byte
typedef unsigned char BYTE;

/// 32-bit word
typedef uint32_t  WORD;

/// Structure representing a job, including data, size, digest, and filename
typedef struct JOB {
	BYTE * data;
	unsigned long long size;
	BYTE digest[64];
	char fname[128];
} JOB;

/// Context structure for SHA-256 hashing, including data buffer, datalen, bitlen, and state
typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} SHA256_CTX;


/*********************** FUNCTION DECLARATIONS **********************/
/// Convert hash buffer to string
char * hash_to_string(BYTE * buff);

/// Print a job structure
void print_job(JOB * j);

/// Print an array of job structures
void print_jobs(JOB ** jobs, int n);

/// Device function to copy 12 bytes
__device__ void mycpy12(uint32_t *d, const uint32_t *s);

/// Device function to copy 16 bytes
__device__ void mycpy16(uint32_t *d, const uint32_t *s);

/// Device function to copy 32 bytes
__device__ void mycpy32(uint32_t *d, const uint32_t *s);

/// Device function to copy 44 bytes
__device__ void mycpy44(uint32_t *d, const uint32_t *s);

/// Device function to copy 48 bytes
__device__ void mycpy48(uint32_t *d, const uint32_t *s);

/// Device function to copy 64 bytes
__device__ void mycpy64(uint32_t *d, const uint32_t *s);

/// Device function for SHA-256 transformation
__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[]);

/// Device function to initialize SHA-256 context
__device__ void sha256_init(SHA256_CTX *ctx);

/// Device function to update SHA-256 context with data
__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);

/// Device function to finalize SHA-256 hashing
__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[]);


#endif   // SHA256_CUH
