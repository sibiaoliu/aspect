/*
  Copyright (C) 2022 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file LICENSE.  If not see
 <http://www.gnu.org/licenses/>.
 */

#include <aspect/particle/property/crystal_preferred_orientation.h>
#include <aspect/geometry_model/interface.h>
#include <world_builder/grains.h>
#include <world_builder/world.h>

#include <aspect/utilities.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {

      template <int dim>
      CrystalPreferredOrientation<dim>::CrystalPreferredOrientation ()
      {
        permutation_operator_3d[0][1][2]  = 1;
        permutation_operator_3d[1][2][0]  = 1;
        permutation_operator_3d[2][0][1]  = 1;
        permutation_operator_3d[0][2][1]  = -1;
        permutation_operator_3d[1][0][2]  = -1;
        permutation_operator_3d[2][1][0]  = -1;
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::initialize ()
      {
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::compute_random_rotation_matrix(Tensor<2,3> &rotation_matrix) const
      {

        // This function is based on an article in Graphic Gems III, written by James Arvo, Cornell University (p 116-120).
        // The original code can be found on  http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        // and is licenced according to this website with the following licence:
        //
        // "The Graphics Gems code is copyright-protected. In other words, you cannot claim the text of the code as your own and
        // resell it. Using the code is permitted in any program, product, or library, non-commercial or commercial. Giving credit
        // is not required, though is a nice gesture. The code comes as-is, and if there are any flaws or problems with any Gems
        // code, nobody involved with Gems - authors, editors, publishers, or webmasters - are to be held responsible. Basically,
        // don't be a jerk, and remember that anything free comes with no guarantee.""
        //
        // The book states in the preface the following: "As in the first two volumes, all of the C and C++ code in this book is in
        // the public domain, and is yours to study, modify, and use."

        // first generate three random numbers between 0 and 1 and multiply them with 2 PI or 2 for z. Note that these are not the same as phi_1, theta and phi_2.

        boost::random::uniform_real_distribution<double> uniform_distribution(0,1);
        double one = uniform_distribution(this->random_number_generator);
        double two = uniform_distribution(this->random_number_generator);
        double three = uniform_distribution(this->random_number_generator);

        double theta = 2.0 * M_PI * one; // Rotation about the pole (Z)
        double phi = 2.0 * M_PI * two; // For direction of pole deflection.
        double z = 2.0* three; //For magnitude of pole deflection.

        // Compute a vector V used for distributing points over the sphere
        // via the reflection I - V Transpose(V).  This formulation of V
        // will guarantee that if x[1] and x[2] are uniformly distributed,
        // the reflected points will be uniform on the sphere.  Note that V
        // has length sqrt(2) to eliminate the 2 in the Householder matrix.

        double r  = std::sqrt( z );
        double Vx = std::sin( phi ) * r;
        double Vy = std::cos( phi ) * r;
        double Vz = std::sqrt( 2.f - z );

        // Compute the row vector S = Transpose(V) * R, where R is a simple
        // rotation by theta about the z-axis.  No need to compute Sz since
        // it's just Vz.

        double st = std::sin( theta );
        double ct = std::cos( theta );
        double Sx = Vx * ct - Vy * st;
        double Sy = Vx * st + Vy * ct;

        // Construct the rotation matrix  ( V Transpose(V) - I ) R, which
        // is equivalent to V S - R.

        rotation_matrix[0][0] = Vx * Sx - ct;
        rotation_matrix[0][1] = Vx * Sy - st;
        rotation_matrix[0][2] = Vx * Vz;
        rotation_matrix[1][0] = Vy * Sx + st;
        rotation_matrix[1][1] = Vy * Sy - ct;
        rotation_matrix[1][2] = Vy * Vz;
        rotation_matrix[2][0] = Vz * Sx;
        rotation_matrix[2][1] = Vz * Sy;
        rotation_matrix[2][2] = 1.0 - z;   // This equals Vz * Vz - 1.0
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::initialize_one_particle_property(const Point<dim> &,
                                                                         std::vector<double> &data) const
      {
        // the layout of the data vector per perticle is the following:
        // 1. M mineral times
        //    1.1  olivine deformation type   -> 1 double, at location
        //                                      => data_position + 0 + mineral_i * (n_grains * 10 + 2)
        //    2.1. Mineral volume fraction    -> 1 double, at location
        //                                      => data_position + 1 + mineral_i *(n_grains * 10 + 2)
        //    2.2. N grains times:
        //         2.1. volume fraction grain -> 1 double, at location:
        //                                      => data_position + 2 + i_grain * 10 + mineral_i *(n_grains * 10 + 2), or
        //                                      => data_position + 2 + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.2. rotation matrix grain -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
        //                                      => data_position + 3 + i_grain * 10 + mineral_i * (n_grains * 10 + 2), or
        //                                      => data_position + 3 + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //
        // Note that we store exactly the same number of grains of all minerals (e.g. olivine and enstatite
        // grains), although their volume fractions may not be the same. We need a minimum amount
        // of grains per tracer to perform reliable statistics on it. This minimum is the same for all phases.
        // and enstatite.
        //
        // Furthermore, for this plugin the following dims are always 3. When using 2D an infinitely thin 3D domain is assumed.
        //
        // The rotation matrix is a direction cosine matrix, representing the orientation of the grain in the domain.
        // The fabric is determined later in the computations, so initialize it to -1.
        std::vector<double> deformation_type(n_minerals, -1.0);
        std::vector<std::vector<double >>volume_fractions_grains(n_minerals);
        std::vector<std::vector<Tensor<2,3>>> rotation_matrices_grains(n_minerals);

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            volume_fractions_grains[mineral_i].resize(n_grains);
            rotation_matrices_grains[mineral_i].resize(n_grains);

            // This will be set by the initial grain subsection.
            bool use_world_builder = false;
            if (use_world_builder)
              {
#ifdef ASPECT_WITH_WORLD_BUILDER
                AssertThrow(false,
                            ExcMessage("Not implemented."))
#else
                AssertThrow(false,
                            ExcMessage("The world builder was requested but not provided. Make sure that aspect is "
                                       "compiled with the World Builder and that you provide a world builder file in the input."))
#endif
              }
            else
              {
                // set volume fraction
                const double initial_volume_fraction = 1.0/n_grains;

                for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
                  {
                    // set volume fraction
                    volume_fractions_grains[mineral_i][grain_i] = initial_volume_fraction;

                    // set a uniform random rotation_matrix per grain
                    this->compute_random_rotation_matrix(rotation_matrices_grains[mineral_i][grain_i]);
                  }
              }
          }

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            data.emplace_back(deformation_type[mineral_i]);
            data.emplace_back(volume_fractions_minerals[mineral_i]);
            for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
              {
                data.emplace_back(volume_fractions_grains[mineral_i][grain_i]);
                for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
                  {
                    const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                    data.emplace_back(rotation_matrices_grains[mineral_i][grain_i][index]);
                  }
              }
          }
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::update_one_particle_property(const unsigned int data_position,
                                                                     const Point<dim> &,
                                                                     const Vector<double> &solution,
                                                                     const std::vector<Tensor<1,dim>> &gradients,
                                                                     const ArrayView<double> &data) const
      {
        // STEP 1: Load data and preprocess it.

        // need access to the pressure, viscosity,
        // get velocity
        Tensor<1,dim> velocity;
        for (unsigned int i = 0; i < dim; ++i)
          velocity[i] = solution[this->introspection().component_indices.velocities[i]];

        // get velocity gradient tensor.
        Tensor<2,dim> velocity_gradient;
        for (unsigned int d=0; d<dim; ++d)
          velocity_gradient[d] = gradients[d];

        // Calculate strain rate from velocity gradients
        const SymmetricTensor<2,dim> strain_rate = symmetrize (velocity_gradient);

        const double dt = this->get_timestep();

        // even in 2d we need 3d strain-rates and velocity gradient tensors. So we make them 3d by
        // adding an extra dimension which is zero.
        SymmetricTensor<2,3> strain_rate_3d;
        strain_rate_3d[0][0] = strain_rate[0][0];
        strain_rate_3d[0][1] = strain_rate[0][1];
        //sym: strain_rate_3d[1][0] = strain_rate[1][0];
        strain_rate_3d[1][1] = strain_rate[1][1];

        if (dim == 3)
          {
            strain_rate_3d[0][2] = strain_rate[0][2];
            strain_rate_3d[1][2] = strain_rate[1][2];
            //sym: strain_rate_3d[2][0] = strain_rate[2][0];
            //sym: strain_rate_3d[2][1] = strain_rate[2][1];
            strain_rate_3d[2][2] = strain_rate[2][2];
          }
        Tensor<2,3> velocity_gradient_3d;
        velocity_gradient_3d[0][0] = velocity_gradient[0][0];
        velocity_gradient_3d[0][1] = velocity_gradient[0][1];
        velocity_gradient_3d[1][0] = velocity_gradient[1][0];
        velocity_gradient_3d[1][1] = velocity_gradient[1][1];
        if (dim == 3)
          {
            velocity_gradient_3d[0][2] = velocity_gradient[0][2];
            velocity_gradient_3d[1][2] = velocity_gradient[1][2];
            velocity_gradient_3d[2][0] = velocity_gradient[2][0];
            velocity_gradient_3d[2][1] = velocity_gradient[2][1];
            velocity_gradient_3d[2][2] = velocity_gradient[2][2];
          }

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {

            set_deformation_type(data_position,data,mineral_i,static_cast<unsigned int>(DeformationType::passive));

            const std::array<double,4> ref_resolved_shear_stress = {{1e60,1e60,1e60,1e60}};


            /**
            * Now we have loaded all the data and can do the actual computation.
            * The computation consists of two parts. The first part is computing
            * the derivatives for the directions and grain sizes. Then those
            * derivatives are used to advect the particle properties.
            */
            double sum_volume_mineral = 0;
            std::pair<std::vector<double>, std::vector<Tensor<2,3>>> derivatives_grains = this->compute_derivatives(data_position,
                                                                                            data,
                                                                                            mineral_i,
                                                                                            strain_rate_3d,
                                                                                            velocity_gradient_3d,
                                                                                            ref_resolved_shear_stress);

            switch (advection_method)
              {
                case AdvectionMethod::forward_euler:

                  sum_volume_mineral = this->advect_forward_euler(data_position,
                                                                  data,
                                                                  mineral_i,
                                                                  dt,
                                                                  derivatives_grains);

                  break;

                case AdvectionMethod::backward_euler:
                  sum_volume_mineral = this->advect_backward_euler(data_position,
                                                                   data,
                                                                   mineral_i,
                                                                   dt,
                                                                   derivatives_grains);

                  break;
              }

            // normalize the volume fractions back to a total of 1 for each mineral
            const double inv_sum_volume_mineral = 1.0/sum_volume_mineral;

            Assert(std::isfinite(inv_sum_volume_mineral),
                   ExcMessage("inv_sum_volume_mineral is not finite. sum_volume_enstatite = "
                              + std::to_string(sum_volume_mineral)));

            for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
              {
                const double volume_fraction_grains = get_volume_fractions_grains(data_position,data,mineral_i,grain_i)*inv_sum_volume_mineral;
                set_volume_fractions_grains(data_position,data,mineral_i,grain_i,volume_fraction_grains);
                Assert(isfinite(get_volume_fractions_grains(data_position,data,mineral_i,grain_i)),
                       ExcMessage("volume_fractions_grains[mineral_i]" + std::to_string(grain_i) + "] is not finite: "
                                  + std::to_string(get_volume_fractions_grains(data_position,data,mineral_i,grain_i)) + ", inv_sum_volume_mineral = "
                                  + std::to_string(inv_sum_volume_mineral) + "."));

                /**
                 * Correct direction cosine matrices numerical error (orthnormality) after integration
                 * Follows same method as in matlab version from Thissen (see https://github.com/cthissen/Drex-MATLAB/)
                 * of finding the nearest orthonormal matrix using the SVD
                 */
                Tensor<2,3> rotation_matrix = get_rotation_matrix_grains(data_position,data,mineral_i,grain_i);
                for (size_t i = 0; i < 3; ++i)
                  {
                    for (size_t j = 0; j < 3; ++j)
                      {
                        Assert(!std::isnan(rotation_matrix[i][j]), ExcMessage("rotation_matrix is nan before orthogonalization."));
                      }
                  }

                rotation_matrix = dealii::project_onto_orthogonal_tensors(rotation_matrix);
                for (size_t i = 0; i < 3; ++i)
                  for (size_t j = 0; j < 3; ++j)
                    {
                      // I don't think this should happen with the projection, but D-Rex
                      // does not do the orthogonal projection, but just clamps the values
                      // to 1 and -1.
                      Assert(std::fabs(rotation_matrix[i][j]) <= 1.0,
                             ExcMessage("The rotation_matrix has a entry larger than 1."));

                      Assert(!std::isnan(rotation_matrix[i][j]),
                             ExcMessage("rotation_matrix is nan after orthoganalization: "
                                        + std::to_string(rotation_matrix[i][j])));

                      Assert(abs(rotation_matrix[i][j]) <= 1.0,
                             ExcMessage("3. rotation_matrix[" + std::to_string(i) + "][" + std::to_string(j) +
                                        "] is larger than one: "
                                        + std::to_string(rotation_matrix[i][j]) + " (" + std::to_string(rotation_matrix[i][j]-1.0) + "). rotation_matrix = \n"
                                        + std::to_string(rotation_matrix[0][0]) + " " + std::to_string(rotation_matrix[0][1]) + " " + std::to_string(rotation_matrix[0][2]) + "\n"
                                        + std::to_string(rotation_matrix[1][0]) + " " + std::to_string(rotation_matrix[1][1]) + " " + std::to_string(rotation_matrix[1][2]) + "\n"
                                        + std::to_string(rotation_matrix[2][0]) + " " + std::to_string(rotation_matrix[2][1]) + " " + std::to_string(rotation_matrix[2][2])));
                    }
              }
          }
      }



      template <int dim>
      UpdateTimeFlags
      CrystalPreferredOrientation<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      InitializationModeForLateParticles
      CrystalPreferredOrientation<dim>::late_initialization_mode () const
      {
        return InitializationModeForLateParticles::interpolate;
      }



      template <int dim>
      UpdateFlags
      CrystalPreferredOrientation<dim>::get_needed_update_flags () const
      {
        return update_values | update_gradients;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      CrystalPreferredOrientation<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " type",1));
            property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " volume fraction",1));

            for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
              {
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " volume fraction",1));

                for (unsigned int index = 0; index < Tensor<2,3>::n_independent_components; ++index)
                  {
                    property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " rotation_matrix " + std::to_string(index),1));
                  }
              }
          }

        return property_information;
      }



      template <int dim>
      double
      CrystalPreferredOrientation<dim>::advect_forward_euler(const unsigned int cpo_index,
                                                             const ArrayView<double> &data,
                                                             const unsigned int mineral_i,
                                                             const double dt,
                                                             const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const
      {
        double sum_volume_fractions = 0;
        Tensor<2,3> rotation_matrix;
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Do the volume fraction of the grain
            Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite before it is set."));
            double volume_fraction_grains = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            volume_fraction_grains =  volume_fraction_grains + dt * volume_fraction_grains * derivatives.first[grain_i];
            set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i, volume_fraction_grains);
            Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                   + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                   + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

            sum_volume_fractions += get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);

            // Do the rotation matrix for this grain
            rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            rotation_matrix += dt * rotation_matrix * derivatives.second[grain_i];
            set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,rotation_matrix);
          }

        Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
        return sum_volume_fractions;
      }



      template <int dim>
      double
      CrystalPreferredOrientation<dim>::advect_backward_euler(const unsigned int cpo_index,
                                                              const ArrayView<double> &data,
                                                              const unsigned int mineral_i,
                                                              const double dt,
                                                              const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const
      {
        double sum_volume_fractions = 0;
        Tensor<2,3> cosine_ref;
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Do the volume fraction of the grain
            double vf_old = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            double vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set."));
            for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
              {
                Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set. grain_i = "
                                                        + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                                                        + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

                vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) + dt * vf_new * derivatives.first[grain_i];

                Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                       + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                       + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));
                if (std::fabs(vf_new-vf_old) < property_advection_tolerance)
                  {
                    break;
                  }
                vf_old = vf_new;
              }

            set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i,vf_new);
            sum_volume_fractions += vf_new;

            // Do the rotation matrix for this grain
            cosine_ref = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            Tensor<2,3> cosine_old = cosine_ref;
            Tensor<2,3> cosine_new = cosine_ref;

            for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
              {
                cosine_new = cosine_ref + dt * cosine_new * derivatives.second[grain_i];

                if ((cosine_new-cosine_old).norm() < property_advection_tolerance)
                  {
                    break;
                  }
                cosine_old = cosine_new;
              }

            set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,cosine_new);

          }


        Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
        return sum_volume_fractions;
      }



      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives(const unsigned int /*cpo_index*/,
                                                            const ArrayView<double> &/*data*/,
                                                            const unsigned int /*mineral_i*/,
                                                            const SymmetricTensor<2,3> & /*strain_rate_3d*/,
                                                            const Tensor<2,3> &velocity_gradient_tensor,
                                                            const std::array<double,4> & /*ref_resolved_shear_stress*/) const
      {
        std::pair<std::vector<double>, std::vector<Tensor<2,3>>> derivatives;
        switch (cpo_derivative_algorithm)
          {
            case CPODerivativeAlgorithm::spin_tensor:
            {
              return compute_derivatives_spin_tensor(velocity_gradient_tensor);
              break;
            }
            default:
              AssertThrow(false, ExcMessage("Internal error."));
              break;
          }
        AssertThrow(false, ExcMessage("Internal error."));
        return derivatives;
      }



      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives_spin_tensor(const Tensor<2,3> &velocity_gradient_tensor) const
      {
        // dA/dt = W * A, where W is the spin tensor and A is the rotation matrix
        // The spin tensor is defined as W = 0.5 * ( L - L^T ), where L is the velocity gradient tensor.
        const Tensor<2,3> spin_tensor = -0.5 *(velocity_gradient_tensor - dealii::transpose(velocity_gradient_tensor));

        return std::pair<std::vector<double>, std::vector<Tensor<2,3>>>(std::vector<double>(n_grains,0.0), std::vector<Tensor<2,3>>(n_grains, spin_tensor));
      }



      template<int dim>
      unsigned int
      CrystalPreferredOrientation<dim>::get_number_of_grains() const
      {
        return n_grains;
      }



      template<int dim>
      unsigned int
      CrystalPreferredOrientation<dim>::get_number_of_minerals() const
      {
        return n_minerals;
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("Crystal Preferred Orientation");
            {
              prm.declare_entry ("Random number seed", "1",
                                 Patterns::Integer (0),
                                 "The seed used to generate random numbers. This will make sure that "
                                 "results are reproducible as long as the problem is run with the "
                                 "same number of MPI processes. It is implemented as final seed = "
                                 "user seed + MPI Rank. ");

              prm.declare_entry ("Number of grains per particle", "50",
                                 Patterns::Integer (1),
                                 "The number of grains of each different mineral "
                                 "each particle contains.");

              prm.declare_entry ("Property advection method", "Backward Euler",
                                 Patterns::Anything(),
                                 "Options: Forward Euler, Backward Euler");

              prm.declare_entry ("Property advection tolerance", "1e-10",
                                 Patterns::Double(0),
                                 "The Backward Euler property advection method involve internal iterations. "
                                 "This option allows for setting a tolerance. When the norm of tensor new - tensor old is "
                                 "smaller than this tolerance, the iteration is stopped.");

              prm.declare_entry ("Property advection max iterations", "100",
                                 Patterns::Integer(0),
                                 "The Backward Euler property advection method involve internal iterations. "
                                 "This option allows for setting the maximum number of iterations. Note that when the iteration "
                                 "is ended by the max iteration amount an assert is thrown.");

              prm.declare_entry ("CPO derivatives algorithm", "Spin tensor",
                                 Patterns::List(Patterns::Anything()),
                                 "Options: Spin tensor");

              prm.enter_subsection("Initial grains");
              {
                prm.declare_entry("Model name","Uniform grains and random uniform rotations",
                                  Patterns::Anything(),
                                  "The model used to initialize the CPO for all particles. "
                                  "Currently 'Uniform grains and random uniform rotations' is the only valid option.");
                prm.enter_subsection("Uniform grains and random uniform rotations");
                {
                  prm.declare_entry ("Minerals", "Olivine: Karato 2008, Enstatite",
                                     Patterns::List(Patterns::Anything()),
                                     "This determines what minerals and fabrics or fabric selectors are used used for the LPO calculation. "
                                     "The options are Olivine: Passive, A-fabric, Olivine: B-fabric, Olivine: C-fabric, Olivine: D-fabric, "
                                     "Olivine: E-fabric, Olivine: Karato 2008 or Enstatite. Passive sets all RRSS entries to the maximum. The "
                                     "Karato 2008 selector selects a fabric based on stress and water content as defined in "
                                     "figure 4 of the Karato 2008 review paper (doi: 10.1146/annurev.earth.36.031207.124120).");


                  prm.declare_entry ("Volume fractions minerals", "0.5, 0.5",
                                     Patterns::List(Patterns::Double(0)),
                                     "The volume fractions for the different minerals. "
                                     "There need to be the same number of values as there are minerals."
                                     "Note that the currently implemented scheme is incompressible and "
                                     "does not allow chemical interaction or the formation of new phases");
                }
                prm.leave_subsection ();
              }
              prm.leave_subsection ();
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::parse_parameters (ParameterHandler &prm)
      {
        AssertThrow(dim == 3, ExcMessage("CPO computations are currently only supported for 3D models. "
                                         "2D computations will work when this assert is removed, but you will need to make sure that the "
                                         "correct 3D strain-rate and velocity gradient tensors are provided to the algorithm."));

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("Crystal Preferred Orientation");
            {
              random_number_seed = prm.get_integer ("Random number seed");
              n_grains = prm.get_integer("Number of grains per particle");

              property_advection_tolerance = prm.get_double("Property advection tolerance");
              property_advection_max_iterations = prm.get_integer ("Property advection max iterations");

              const std::string temp_cpo_derivative_algorithm = prm.get("CPO derivatives algorithm");

              if (temp_cpo_derivative_algorithm == "Spin tensor")
                {
                  cpo_derivative_algorithm = CPODerivativeAlgorithm::spin_tensor;
                }
              else if (temp_cpo_derivative_algorithm ==  "D-Rex 2004")
                {
                  AssertThrow(false,
                              ExcMessage("D-Rex 2004 has not been implemented yet."))
                }
              else
                {
                  AssertThrow(false,
                              ExcMessage("The CPO derivatives algorithm needs to be one of the following: "
                                         "Spin tensor, D-Rex 2004."))
                }

              const std::string temp_advection_method = prm.get("Property advection method");
              if (temp_advection_method == "Forward Euler")
                {
                  advection_method = AdvectionMethod::forward_euler;
                }
              else if (temp_advection_method == "Backward Euler")
                {
                  advection_method = AdvectionMethod::backward_euler;
                }
              else
                {
                  AssertThrow(false, ExcMessage("particle property advection method not found: \"" + temp_advection_method + "\""));
                }

              prm.enter_subsection("Initial grains");
              {
                const std::string model_name = prm.get("Model name");
                AssertThrow(model_name == "Uniform grains and random uniform rotations",
                            ExcMessage("No model named " + model_name + "for CPO particle property initialization. "
                                       + "Only the model \"Uniform grains and random uniform rotations\" is available."));
                prm.enter_subsection("Uniform grains and random uniform rotations");
                {
                  const std::vector<std::string> temp_deformation_type_selector = dealii::Utilities::split_string_list(prm.get("Minerals"));
                  n_minerals = temp_deformation_type_selector.size();
                  deformation_type_selector.resize(n_minerals);

                  for (size_t mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
                    {
                      if (temp_deformation_type_selector[mineral_i] == "Passive")
                        {
                          deformation_type_selector[mineral_i] = DeformationTypeSelector::passive;
                        }
                      else
                        {
                          AssertThrow(false,
                                      ExcMessage("The fabric needs to be assigned one of the following comma-delimited values: Olivine: Karato 2008, "
                                                 "Olivine: A-fabric, Olivine: B-fabric, Olivine: C-fabric, Olivine: D-fabric,"
                                                 "Olivine: E-fabric, Enstatite, Passive."))
                        }
                    }

                  volume_fractions_minerals = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Volume fractions minerals")));
                  double volume_fractions_minerals_sum = 0;
                  for (auto fraction : volume_fractions_minerals)
                    {
                      volume_fractions_minerals_sum += fraction;
                    }

                  AssertThrow(abs(volume_fractions_minerals_sum-1.0) < 2.0 * std::numeric_limits<double>::epsilon(),
                              ExcMessage("The sum of the CPO volume fractions should be one."));
                }
                prm.leave_subsection();
              }
              prm.leave_subsection();
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(CrystalPreferredOrientation,
                                        "crystal preferred orientation",
                                        "The plugin manages and computes the evolution of Lattice/Crystal Preferred Orientations (LPO/CPO) "
                                        "on particles. Each ASPECT particle can be assigned many grains. Each grain is assigned a size and a orientation "
                                        "matrix. This allows for LPO evolution tracking with polycrystalline kinematic CrystalPreferredOrientation evolution models such "
                                        "as D-Rex (Kaminski and Ribe, 2001; Kaminski et al., 2004).")
    }
  }
}
