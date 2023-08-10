#ifndef SHA256_CUH
#define SHA256_CUH

#include <stdint.h>

/****************************** MACROS ******************************/
#define SHA256_DIGEST_LENGTH 32
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
        printf("GPU: cudaError %d (%s)\n", err, cudaGetErrorString(err)); \
}


/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef uint32_t  WORD;             // 32-bit word, change to "long" for 16-bit machines

typedef struct JOB {
	BYTE * data;
	unsigned long long size;
	BYTE digest[64];
	char fname[128];
}JOB;

typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} SHA256_CTX;


/*********************** FUNCTION DECLARATIONS **********************/
// char * print_sha(BYTE * buff);
char * hash_to_string(BYTE * buff);
void print_job(JOB * j);
void print_jobs(JOB ** jobs, int n);
__device__ void mycpy12(uint32_t *d, const uint32_t *s);
__device__ void mycpy16(uint32_t *d, const uint32_t *s);
__device__ void mycpy32(uint32_t *d, const uint32_t *s);
__device__ void mycpy44(uint32_t *d, const uint32_t *s);
__device__ void mycpy48(uint32_t *d, const uint32_t *s);
__device__ void mycpy64(uint32_t *d, const uint32_t *s);
__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[]);
__device__ void sha256_init(SHA256_CTX *ctx);
__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[]);


#endif   // SHA256_CUH
