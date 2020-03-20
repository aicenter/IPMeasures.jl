# the following functions are adapted from CUDAnative.jl/examples/pairwise.jl

# pairwise distance calculation kernel
function pairwise_dist_kernel(lat::CuDeviceVector{Float32}, lon::CuDeviceVector{Float32},
                              rowresult::CuDeviceMatrix{Float32}, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        # store to shared memory
        shmem = @cuDynamicSharedMem(Float32, 2*blockDim().x + 2*blockDim().y)
        if threadIdx().y == 1
            shmem[threadIdx().x] = lat[i]
            shmem[blockDim().x + threadIdx().x] = lon[i]
        end
        if threadIdx().x == 1
            shmem[2*blockDim().x + threadIdx().y] = lat[j]
            shmem[2*blockDim().x + blockDim().y + threadIdx().y] = lon[j]
        end
        sync_threads()

        # load from shared memory
        lat_i = shmem[threadIdx().x]
        lon_i = shmem[blockDim().x + threadIdx().x]
        lat_j = shmem[2*blockDim().x + threadIdx().y]
        lon_j = shmem[2*blockDim().x + blockDim().y + threadIdx().y]

        @inbounds rowresult[i, j] = haversine_gpu(lat_i, lon_i, lat_j, lon_j, 6372.8f0)
    end

    return
end

function pairwise_dist_gpu(lat::CuVector, lon::CuVector)
    # allocate
    n = length(lat)
    rowresult_gpu = CuArray(zeros(Float32, n, n))

    # calculate launch configuration
    function get_config(kernel)
        # calculate a 2D block size from the suggested 1D configuration
        # NOTE: we want our launch configuration to be as square as possible,
        #       because that minimizes shared memory usage
        function get_threads(threads)
            threads_x = floor(Int, sqrt(threads))
            threads_y = threads ÷ threads_x
            return (threads_x, threads_y)
        end

        # calculate the amount of dynamic shared memory for a 2D block size
        get_shmem(threads) = 2 * sum(threads) * sizeof(Float32)

        fun = kernel.fun
        config = launch_configuration(fun, shmem=threads->get_shmem(get_threads(threads)))

        # convert to 2D block size and figure out appropriate grid size
        threads = get_threads(config.threads)
        blocks = ceil.(Int, n ./ threads)
        shmem = get_shmem(threads)

        return (threads=threads, blocks=blocks, shmem=shmem)
    end

    @cuda config=get_config pairwise_dist_kernel(lat_gpu, lon_gpu, rowresult_gpu, n)

    return rowresult_gpu
end
