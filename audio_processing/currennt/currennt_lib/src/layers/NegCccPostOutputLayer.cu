/******************************************************************************
 * Copyright (c) 2014 Felix Weninger
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "NegCccPostOutputLayer.hpp"
#include "../helpers/getRawPointer.cuh"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    // "Covariance" per sequence and feature
    // q = q + targets(t) * output(t) - m * output(t)
    // TODO: respect dummies
    struct ComputeQFn
    {
        int layerSize; // number of targets / output features
        const char *patTypes;
        
        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, real_t, real_t, int> &values) const
        {
            int outputIdx = values.get<4>();
            int patIdx = outputIdx / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE)
                return values.get<0>();

            real_t output = values.get<2>();
            return values.get<0>() + values.get<1>() * output
                                   - values.get<3>() * output;
            //return values.get<0>() + (values.get<1>() - values.get<3>()) * output;
        }
    };
    
    struct SumToMeanFn
    {
        __host__ __device__ real_t operator() (real_t x, int n) const
        {
            if (n == 0)
                return real_t(0.0);
            else
                return x / real_t(n);
        }
    };
    
    // SSEs per sequence and feature
    // s = s + (target(t) - output(t)) ^ 2
    struct ComputeSseFn
    {
        int layerSize; // number of targets / output features
        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, real_t, int> &values) const
        {
            // unpack the tuple
            real_t target = values.get<1>();
            real_t output = values.get<2>();
            int outputIdx = values.get<3>();

            // check if we have to skip this value
            int patIdx = outputIdx / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE)
                return values.get<0>();

            // calculate the error
            real_t diff = target - output;
            return values.get<0>() + (diff * diff);
        }
    };
    
    // TODO implement this more efficiently
    struct CountTimestepsFn
    {
        int layerSize;
        const char *patTypes;
        
        __host__ __device__ int operator() (const thrust::tuple<int, int> &values) const
        {
            int outputIdx = values.get<1>();
            int patIdx = outputIdx / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE)
                return values.get<0>();
            else
                return values.get<0>() + 1;
        }        
    };

    // Compute CCC from "covariance" and SSE
    struct ComputeNegCccFn
    {
        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t> &values) const
        {
            real_t q = values.get<0>();
            real_t s = values.get<1>();
            if (q == 0 && s == 0)
                return 0;
            return -q / (0.5 * s + q);
        }
    };

    // Compute gradient of CCC using previously calculated "covariance" and SSE
    struct ComputeDeltaNegCccFn
    {
        int layerSize;
        int nccc;

        const char *patTypes;
        real_t *Q;
        real_t *S;
        real_t *target_means;

        __host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t output    = t.get<0>();
            real_t target    = t.get<1>();
            int    outputIdx = t.get<2>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if the pattern is a dummy
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;
                
            int cccIdx = outputIdx % nccc;

            real_t ss = 0.5 * S[cccIdx];            
            real_t delta = ((target - target_means[cccIdx]) * ss
                           - (output - target) * Q[cccIdx])
                           / ((ss + Q[cccIdx]) * (ss + Q[cccIdx]));

            return -delta;
        }
    };
    
} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice>
    NegCccPostOutputLayer<TDevice>::NegCccPostOutputLayer(const helpers::JsonValue &layerChild, Layer<TDevice> &precedingLayer)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, precedingLayer.size())
    {
        // useTotalCCC?
    }

    template <typename TDevice>
    NegCccPostOutputLayer<TDevice>::~NegCccPostOutputLayer()
    {
    }

    template <typename TDevice>
    const std::string& NegCccPostOutputLayer<TDevice>::type() const
    {
        static const std::string s("neg_ccc");
        return s;
    }

    template <typename TDevice>
    real_t NegCccPostOutputLayer<TDevice>::calculateError()
    {
        // sum up CCCs per feature and sequence
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
        real_t totalCCC = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(
                _Q.begin(),
                _S.begin()
            )),
            thrust::make_zip_iterator(thrust::make_tuple(
                _Q.end(),
                _S.end()
            )),
            internal::ComputeNegCccFn(),
            (real_t)0,
            thrust::plus<real_t>()
        );
        /*
        std::cout << "CCC = " << totalCCC << std::endl;
        */
        return totalCCC;
    }

    template <typename TDevice>
    void NegCccPostOutputLayer<TDevice>::computeForwardPass()
    {
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
        int nccc = this->size() * this->parallelSequences();
        
        _Q.resize(nccc);
        _S.resize(nccc);
        _target_means.resize(nccc);
        thrust::fill(_Q.begin(), _Q.end(), 0);
        thrust::fill(_S.begin(), _S.end(), 0);
        thrust::fill(_target_means.begin(), _target_means.end(), 0);
        
        /*
        std::cout << "targets: " ;
        thrust::copy(this->_targets().begin(), this->_targets().end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        std::cout << "outputs: " ;
        thrust::copy(this->_actualOutputs().begin(), this->_actualOutputs().end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        */
        
        typename TDevice::int_vector seqLens(this->parallelSequences() * this->size());
        internal::CountTimestepsFn cfn;
        cfn.layerSize = this->size();
        cfn.patTypes  = helpers::getRawPointer(this->patTypes());
        for (int t = 0; t < this->curMaxSeqLength(); ++t) {
            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(
                    seqLens.begin(), 
                    thrust::counting_iterator<int>(0) + t * nccc
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    seqLens.end(), 
                    thrust::counting_iterator<int>(0) + (t + 1) * nccc
                )),
                seqLens.begin(),
                cfn
            );
        }
        // thrust::fill(seqLens.begin(), seqLens.end(), this->curMaxSeqLength());
        /*
        std::cout << "seq lens: ";
        thrust::copy(seqLens.begin(), seqLens.end(), std::ostream_iterator<float>(std::cout, " "));

        std::cout << std::endl;
        */
        
        for (int t = 0; t < this->curMaxSeqLength(); ++t) {
            thrust::transform(
                this->_targets().begin() + t * nccc,
                this->_targets().begin() + (t + 1) * nccc,
                _target_means.begin(),
                _target_means.begin(),
                thrust::plus<real_t>()
            );
        }
        
        thrust::transform(
            _target_means.begin(),
            _target_means.end(),
            seqLens.begin(),
            _target_means.begin(),
            internal::SumToMeanFn()
//            thrust::divides<real_t>()
        );
        
        /*
        std::cout << "target means: ";
        thrust::copy(_target_means.begin(), _target_means.end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        */
            
        internal::ComputeSseFn sfn;
        sfn.layerSize = this->size();
        sfn.patTypes  = helpers::getRawPointer(this->patTypes());
        
        internal::ComputeQFn qfn;
        qfn.layerSize = this->size();
        qfn.patTypes  = helpers::getRawPointer(this->patTypes());
        
        for (int t = 0; t < this->curMaxSeqLength(); ++t) {
            // quaternary op: q = q + targets(t) * output(t) - m * output(t)
            // all are F x B matrices in local storage
            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(
                    _Q.begin(), 
                    this->_targets().begin() + t * nccc,
                    this->_actualOutputs().begin() + t * nccc,
                    _target_means.begin(),
                    thrust::counting_iterator<int>(0) + t * nccc
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    _Q.end(), 
                    this->_targets().begin() + (t + 1) * nccc,
                    this->_actualOutputs().begin() + (t + 1) * nccc,
                    _target_means.end(),
                    thrust::counting_iterator<int>(0) + (t + 1) * nccc
                )),
                _Q.begin(),
                qfn
            );
            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(
                    _S.begin(), 
                    this->_targets().begin() + t * nccc,
                    this->_actualOutputs().begin() + t * nccc,
                    thrust::counting_iterator<int>(0) + t * nccc
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    _S.end(), 
                    this->_targets().begin() + (t + 1) * nccc,
                    this->_actualOutputs().begin() + (t + 1) * nccc,
                    thrust::counting_iterator<int>(0) + (t + 1) * nccc
                )),
                _S.begin(),
                sfn
            );
        }
        
        
        /*
        std::cout << "Q: ";
        thrust::copy(_Q.begin(), _Q.end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        
        std::cout << "S: ";
        thrust::copy(_S.begin(), _S.end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        */

        // if total cc (across all sequences in the batch) is desired, sum up Q and S
        
    }

    template <typename TDevice>
    void NegCccPostOutputLayer<TDevice>::computeBackwardPass()
    {
        // calculate the errors
        internal::ComputeDeltaNegCccFn fn;
        fn.layerSize = this->size();
        fn.patTypes  = helpers::getRawPointer(this->patTypes());
        fn.nccc = this->parallelSequences() * this->size();
        fn.Q = helpers::getRawPointer(_Q);
        fn.S = helpers::getRawPointer(_S);
        fn.target_means = helpers::getRawPointer(_target_means);
        
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(this->_actualOutputs().begin(),   this->_targets().begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->_actualOutputs().begin()+n, this->_targets().begin()+n, thrust::counting_iterator<int>(0)+n)),
            this->_outputErrors().begin(),
            fn
            );
            
        /*
        std::cout << "Output gradients: ";
        thrust::copy(this->_outputErrors().begin(), this->_outputErrors().end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        */
    }


    // explicit template instantiations
    template class NegCccPostOutputLayer<Cpu>;
    template class NegCccPostOutputLayer<Gpu>;

} // namespace layers
