/**
 * kinect_shm_server.cpp
 *
 * libfreenect2 でRGB + Depth フレームを取得し、
 * POSIX 共有メモリに書き込む固定サーバー。
 * Python 側は共有メモリを読み取り MediaPipe に渡す。
 *
 * 共有メモリレイアウト:
 *   [0..3]        : uint32_t  シーケンス番号 (フレームごとにインクリメント)
 *   [4..7]        : uint32_t  フラグ (1=有効データあり, 0=初期化中)
 *   [8..2073607]  : uint8_t   RGB  1920x1080x3  = 6,220,800 bytes  (BGR順)
 *   [2073608..2514231]: float  Depth 512x424x1  = 217,088 floats = 868,352 bytes (mm)
 *   Total: ~7.1 MB

[ How to compile ]
 *
 * g++ -O2 -o kinect_shm_server kinect_shm_server.cpp `pkg-config freenect2 --libs --cflags` 
 */

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <atomic>

// --- 共有メモリ定数 ---
static const char* SHM_NAME    = "/kinect2_frames";
static const int   RGB_W       = 1920;
static const int   RGB_H       = 1080;
static const int   DEP_W       = 512;
static const int   DEP_H       = 424;
static const size_t RGB_BYTES  = RGB_W * RGB_H * 3;           // 6,220,800
static const size_t DEP_BYTES  = DEP_W * DEP_H * sizeof(float); // 868,352
static const size_t HDR_BYTES  = 8;                            // seq(4) + flag(4)
static const size_t SHM_SIZE   = HDR_BYTES + RGB_BYTES + DEP_BYTES;

static const size_t OFF_SEQ    = 0;
static const size_t OFF_FLAG   = 4;
static const size_t OFF_RGB    = HDR_BYTES;
static const size_t OFF_DEP    = HDR_BYTES + RGB_BYTES;

// --- 終了フラグ ---
static std::atomic<bool> g_running(true);

void sighandler(int) { g_running = false; }

int main(int argc, char* argv[]){
    
    signal(SIGINT,  sighandler);
    signal(SIGTERM, sighandler);

    // ---- 共有メモリ作成 ----
    shm_unlink(SHM_NAME); // 残骸を削除
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) { perror("shm_open"); return 1; }
    if (ftruncate(shm_fd, SHM_SIZE) < 0) { perror("ftruncate"); return 1; }

    uint8_t* shm = static_cast<uint8_t*>(
        mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0) );
    if (shm == MAP_FAILED) { perror("mmap"); return 1; }
    memset(shm, 0, SHM_SIZE);

    uint32_t* seq_ptr  = reinterpret_cast<uint32_t*>(shm + OFF_SEQ);
    uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(shm + OFF_FLAG);
    uint8_t*  rgb_ptr  = shm + OFF_RGB;
    float*    dep_ptr  = reinterpret_cast<float*>(shm + OFF_DEP);

    // ---- libfreenect2 初期化 ----
    libfreenect2::Freenect2 freenect2;
    if (freenect2.enumerateDevices() == 0) {
        fprintf(stderr, "[kinect_shm] No Kinect device found.\n");
        return 1;
    }

    // CPU パイプライン (OpenGL は表示不要なので CPU を使用)
//    libfreenect2::PacketPipeline* pipeline = new libfreenect2::CpuPacketPipeline();
    libfreenect2::PacketPipeline* pipeline = new libfreenect2::OpenGLPacketPipeline();
    libfreenect2::Freenect2Device* device = freenect2.openDefaultDevice(pipeline);
    if (!device) {
        fprintf(stderr, "[kinect_shm] Failed to open device.\n");
        return 1;
    }

    // Color + Depth のみ受け取る
    int types = libfreenect2::Frame::Color | libfreenect2::Frame::Depth;
    libfreenect2::SyncMultiFrameListener listener(types);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);

    if (!device->start()) {
        fprintf(stderr, "[kinect_shm] Failed to start device.\n");
        return 1;
    }

    fprintf(stdout, "[kinect_shm] Device started. Serial: %s  Firmware: %s\n",
            device->getSerialNumber().c_str(),
            device->getFirmwareVersion().c_str());

    // IRカメラパラメータをファイルに出力 (Python側で読む)
    {
        libfreenect2::Freenect2Device::IrCameraParams ir = device->getIrCameraParams();
        FILE* fp = fopen("/tmp/kinect2_ir_params.txt", "w");
        if (fp) {
            fprintf(fp, "fx=%f\nfy=%f\ncx=%f\ncy=%f\n", ir.fx, ir.fy, ir.cx, ir.cy);
            fclose(fp);
            fprintf(stdout, "[kinect_shm] IR params: fx=%.2f fy=%.2f cx=%.2f cy=%.2f\n",
                    ir.fx, ir.fy, ir.cx, ir.cy);
        }
    }

    libfreenect2::FrameMap frames;
    uint32_t seq = 0;

    fprintf(stdout, "[kinect_shm] Running. Press Ctrl+C to stop.\n");
    fflush(stdout);

    while (g_running)
    {
        if (!listener.waitForNewFrame(frames, 10 /* ms */))
            continue;

        libfreenect2::Frame* color = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];

        // --- RGB コピー (BGRX 4ch → BGR 3ch) ---
        const uint8_t* src = color->data;
        for (int i = 0, j = 0; i < RGB_W * RGB_H; ++i, src += 4, j += 3) {
            rgb_ptr[j]   = src[2]; // R
            rgb_ptr[j+1] = src[1]; // G
            rgb_ptr[j+2] = src[0]; // B
        }

        // --- Depth コピー (float mm) ---
        memcpy(dep_ptr, depth->data, DEP_W * DEP_H * sizeof(float));

        // --- シーケンス番号更新 (flag=1 で有効を通知) ---
        ++seq;
        *seq_ptr  = seq;
        *flag_ptr = 1u;

        listener.release(frames);
    }

    fprintf(stdout, "\n[kinect_shm] Shutting down...\n");
    *flag_ptr = 0u;
    device->stop();
    device->close();
    munmap(shm, SHM_SIZE);
    close(shm_fd);
    shm_unlink(SHM_NAME);
    return 0;
}

