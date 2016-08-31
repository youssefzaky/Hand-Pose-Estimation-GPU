///////////////////////////////////////////////////////////////////////////////
// CUDAHostTimer: High-performance host timer for CUDA
// Copyright (C) 2013 Ashwin Nanjappa
// Released under the MIT License
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined( _WIN32 )

#include <windows.h>

class WinCudaHostTimer
{
private:
	float         freq;
	LARGE_INTEGER startTime;
	LARGE_INTEGER endTime;

public:
    WinCudaHostTimer()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        freq = 1.0f / freq.QuadPart;
        return;
    }

    void start()
    {
        QueryPerformanceCounter(&startTime);
        return;
    }

    void stop()
    {
        QueryPerformanceCounter(&endTime);
        return;
    }

    double value() const
    {
        return (endTime.QuadPart - startTime.QuadPart) * freq;
    }
};

typedef WinCudaHostTimer CudaHostTimer;

#else

#include <sys/time.h>

class LinuxCudaHostTimer
{
private:
    long long startTime;
    long long endTime;

    long long getTime()
    {
        long long time;
        struct timeval tv;

        gettimeofday(&tv, NULL);
        time  = 1000000000LL; // seconds->nanonseconds
        time *= tv.tv_sec;
        time += tv.tv_usec * 1000; // ms->ns

        return time;
    }

public:
    void start()
    {
        cudaDeviceSynchronize();
        startTime = getTime();
    }

    void stop()
    {
        cudaDeviceSynchronize();
        endTime = getTime();
    }

    double value()
    {
        return ((double) endTime - startTime) / 1000000000LL;
    }
};

typedef LinuxCudaHostTimer CudaHostTimer;

#endif

///////////////////////////////////////////////////////////////////////////////
