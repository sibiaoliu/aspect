/*
  Copyright (C) 2021 - 2023 by the authors of the ASPECT code.
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

#include <array>
#include <utility>
#include <limits>

#include <aspect/simulator_access.h>
#include <aspect/simulator.h>
#include <aspect/utilities.h>
#include <aspect/global.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/signaling_nan.h>

#include <aspect/heating_model/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/material_model/visco_plastic.h>

#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/ellipsoidal_chunk.h>

/* Head file for injection term*/
namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * This material model can take any other material model as a base model,
     * and defines a material injection zone via a dilation term applied to
     * the Stokes equations.
     *
     * The method is mainly described in the following paper:
     * @code
     * @article{theissen2011coupled,
     *   title={Coupled mechanical and hydrothermal modeling of crustal
     *          accretion at intermediate to fast spreading ridges},
     *   author={Theissen-Krah, Sonja and Iyer, Karthik and R{\"u}pke, Lars H
     *           and Morgan, Jason Phipps},
     *   journal={Earth and Planetary Science Letters},
     *   volume={311},
     *   number={3-4},
     *   pages={275--286},
     *   year={2011},
     *   publisher={Elsevier}
     * }
     * @endcode
     *
     * @ingroup MaterialModels
     */

    template <int dim>
    class PrescribedDikeInjection : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Initialize the model at the beginning of the run.
         */
        void initialize() override;

        /**
         * Update the base model and dilation function at the beginning
         * of each timestep.
         */
        void update() override;

        /**
         * At the begining of each timestep, randomly relocate the dike
         * (magma-intrusion zone).The location varies as a Gaussian distribution
         * around a user-defined set of line-segments.
         */
        virtual
        double dike_distribution (const Point<dim> &position) const;

        /**
         * Function to compute the material properties in @p out given
         * the inputs in @p in.
         */
        void
        evaluate (const typename Interface<dim>::MaterialModelInputs &in,
                  typename Interface<dim>::MaterialModelOutputs &out) const override;

        /**
         * Declare the parameters through input files.
         */
        static void
        declare_parameters (ParameterHandler &prm);

        /**
         * Parse parameters through the input file
         */
        void
        parse_parameters (ParameterHandler &prm) override;

        /**
         * Indicate whether material is compressible only based on
         * the base model.
         */
        bool is_compressible () const override;

        /**
         * Method to calculate reference viscosity. Not used anymore.
         */
        double reference_viscosity () const override;

        void
        create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const override;

      private:
        /**
         * Parameters of the random white noise function.
         * Whether or not use random white noise.
         */
        bool enable_random_noise;
        
        /**
         * Whether or not the domain is a cartesian box.
         */
        bool cartesian_domain = true;

        /**
         * The value of the seed for the random number generator
         */
        double seed;

        /**
         * The maximum amplitude of the Gaussian amplitude of the noise.
         */
        double A;

        /**
         * The standard deviation of the Gaussian amplitude of the noise.
         */
        double sigma;

        /**
         * The list of line segments consisting of two 2d coordinates per segment (even in 2d).
         * The segments represent the rift axis.
         */        
        std::vector<std::array<Point<2>, 2 > > point_list;
        /**
         * A table with random noise for the
         * second invariant of the strain.
         */        
        std::array<unsigned int,dim> grid_intervals;
        Functions::InterpolatedUniformGridData<dim> *interpolate_noise;

        /**
         * Specify the dike shape, duration.
         */
        double dike_width;
        double dike_depth;
        double dike_duration;

        /**
         * Specify the time-dependent linear function of the melt fraction M.
         */
        double    half_extension_rate;
        double    M_value;
        double    M_cutoff_time1;
        double    M_cutoff_time2;
        double    M_cutoff_value;

        /**
         * Pointer to the material model used as the base model.
         */
        std::unique_ptr<MaterialModel::Interface<dim>> base_model;
    };
  }
}

/* Head file for latent heat term*/
namespace aspect
{
  namespace HeatingModel
  {
    using namespace dealii;

    /**
     * A class that implements the latent heat released during crystallization
     * of the melt lens and heating by melt injection into the model. It takes
     * the amount of material added on the right-hand side of the Stokes equations
     * and adds the corresponding heating term to the energy equation (considering
     * the latent heat of crystallization and the different temperature of the
     * injected melt).
     *
     * @ingroup HeatingModels
     */
    template <int dim>
    class LatentHeatDikeInjection : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Compute the heating model outputs for this class.
         */
        void
        evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
                  const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
                  HeatingModel::HeatingModelOutputs &heating_model_outputs) const override;
        void
        create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &material_model_outputs) const override;
        /**
         * @name Functions used in dealing with run-time parameters
         * @{
         */

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        void parse_parameters (ParameterHandler &prm) override;
        /**
         * @}
         */

      private:
        /**
         * Properties of injected material.
         */
        double latent_heat_of_crystallization;
        double temperature_of_injected_melt;
    };
  }
}

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void
    PrescribedDikeInjection<dim>::initialize()
    {
      base_model->initialize();

      // Add the random noise function
      AssertThrow(Plugins::plugin_type_matches<const PrescribedDikeInjection<dim>>(this->get_material_model()),
                  ExcMessage("This initial condition only works with the material model prescribed dike injection."));

      Point<dim> extents_min, extents_max;
      TableIndices<dim> size_idx;
      for (unsigned int d=0; d<dim; ++d)
        size_idx[d] = grid_intervals[d]+1;

      Table<dim,double> white_noise;
      white_noise.TableBase<dim,double>::reinit(size_idx);
      std::array<std::pair<double,double>,dim> grid_extents;

      //Currently, we only test the box geometry
      if (cartesian_domain)
        {
          const GeometryModel::Box<dim> *geometry_model
            = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model());

          // Min and max of each direction (m)
          extents_min = geometry_model->get_origin();
          extents_max = geometry_model->get_extents() + extents_min;
        }
      else if (const GeometryModel::Chunk<dim> *geometry_model
               = dynamic_cast<const GeometryModel::Chunk<dim> *>(&this->get_geometry_model()))
        {
          // Min and max radius
          extents_max[0] = geometry_model->outer_radius();
          //extents_min[0] = std::max(extents_max[0]-strain_depth-5.*strain_halfwidth,geometry_model->inner_radius());
          extents_max[0] = geometry_model->inner_radius();
          // Min and max longitude (already in radians)
          extents_min[1] = geometry_model->east_longitude();
          extents_max[1] = geometry_model->west_longitude();

          // Min and max latitude (already in radians), convert to colatitude
          if (dim == 3)
            {
              extents_max[dim-1] = 0.5 * numbers::PI - geometry_model->south_latitude();
              extents_min[dim-1] = 0.5 * numbers::PI - geometry_model->north_latitude();
            }
        }
      else if (const GeometryModel::EllipsoidalChunk<dim> *geometry_model
               = dynamic_cast<const GeometryModel::EllipsoidalChunk<dim> *>(&this->get_geometry_model()))
        {
          // Check that the model is not elliptical
          AssertThrow(geometry_model->get_eccentricity() == 0.0, ExcMessage("This boundary velocity plugin cannot be used with a non-zero eccentricity. "));

          // Min and max radius
          extents_max[0] = geometry_model->get_semi_major_axis_a();
          //extents_min[0] = std::max(extents_max[0]-geometry_model->maximal_depth(), extents_max[0]-strain_depth-5.*strain_halfwidth);
          extents_min[0] = extents_max[0];

          // Assume chunk outlines are lat/lon parallel
          std::vector<Point<2> > corners = geometry_model->get_corners();
          // Convert to radians, lon, colat
          extents_min[1] = corners[1][0] * numbers::PI / 180.;
          extents_max[1] = corners[0][0] * numbers::PI / 180.;
          extents_min[dim-1] = 0.5 * numbers::PI - corners[0][1] * numbers::PI / 180.;
          extents_max[dim-1] = 0.5 * numbers::PI - corners[2][1] * numbers::PI / 180.;
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("This initial condition only works with the box or (ellipsoidal) chunk geometry model."));
        }

      for (unsigned int d=0; d<dim; ++d)
        {
          grid_extents[d].first=extents_min[d];
          grid_extents[d].second=extents_max[d];
        }

      // use a fixed number as seed for random generator
      // this is important if we run the code on more than 1 processor
      std::srand(seed);

      TableIndices<dim> idx;

      for (unsigned int i=0; i<white_noise.size()[0]; ++i)
        {
          idx[0] = i;
          for (unsigned int j=0; j<white_noise.size()[1]; ++j)
            {
              idx[1] = j;
              if (dim == 3)
                {
                  for (unsigned int k=0; k<white_noise.size()[dim-1]; ++k)
                    {
                      idx[dim-1] = k;
                      // std::rand will give a value between zero and RAND_MAX (usually INT_MAX).
                      // The modulus of this value and 10000, gives a value between 0 and 10000-1.
                      // Subsequently dividing by 5000.0 will give value between 0 and 2 (excluding 2).
                      // Subtracting 1 will give a range [-1,1)
                      // Because we want values [0,1), we change our white noise computation to:
                      white_noise(idx) = ((std::rand() % 10000) / 10000.0);
                    }
                }
              else
                white_noise(idx) = ((std::rand() % 10000) / 10000.0);
            }
        }

      interpolate_noise = new Functions::InterpolatedUniformGridData<dim> (grid_extents,
                                                                           grid_intervals,
                                                                           white_noise);      
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::update()
    {
      base_model->update();
    }

    template <int dim>
    double
    PrescribedDikeInjection<dim>::
    dike_distribution (const Point<dim> &position) const
    {
      // Initiate distance with large value
      double distance_to_dike_axis = 1e23;
      double temp_distance = 0;

      // For spherical geometries we need to reorder the coordinates
      Point<dim> natural_coords = position;

      // Loop over all line segments
      for (unsigned int i_segments = 0; i_segments < point_list.size(); ++i_segments)
        {
          if (cartesian_domain)
            {
              if (dim == 2)
                temp_distance = std::abs(natural_coords[0]-point_list[i_segments][0][0]);
              else
                {
                  // Get the surface coordinates by dropping the last coordinate
                  const Point<2> surface_position = Point<2>(natural_coords[0],natural_coords[1]);
                  temp_distance = std::abs(Utilities::distance_to_line(point_list[i_segments], surface_position));
                }
            }
          // chunk (spherical) geometries
          else
            {
              // spherical coordinates in radius [m], lon [rad], colat [rad] format
              const std::array<double,dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
              natural_coords[0] = spherical_point[0];
              Point<2> surface_position;
              for (unsigned int d=0; d<dim-1; ++d)
                {
                  surface_position[d] = spherical_point[d+1];
                  natural_coords[d+1] = spherical_point[d+1];
                }

              temp_distance = (dim == 2) ? std::abs(surface_position[0]-point_list[i_segments][0][0]) : Utilities::distance_to_line(point_list[i_segments], surface_position);
            }

          // Get the minimum distance
          distance_to_dike_axis = std::min(distance_to_dike_axis, temp_distance);
        }

      // Smoothing of noise with lateral distance to the dike axis
      const double noise_amplitude = A * std::exp((-std::pow(distance_to_dike_axis,2)/(2.0*std::pow(sigma,2))));
      // Add randomness
      if (enable_random_noise == true)
        return noise_amplitude * interpolate_noise->value(natural_coords);
      else
        return noise_amplitude;

    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::evaluate(const typename Interface<dim>::MaterialModelInputs &in,
                                      typename Interface<dim>::MaterialModelOutputs &out) const
    {
      // fill variable out with the results form the base material model
      // The base model may have additional outputs such as frictional angle
      // in visco-plastic model, so we need to copy material properties first.
      typename Interface<dim>::MaterialModelOutputs base_output(out.n_evaluation_points(),
                                                                this->introspection().n_compositional_fields);

      // Move the additional outputs to base_output so that our models can fill
      // them if desired:
      base_output.move_additional_outputs_from(out);
      base_model->evaluate(in, base_output);

      // Copy required properties
      out.viscosities = base_output.viscosities;
      out.densities = base_output.densities;
      out.thermal_expansion_coefficients = base_output.thermal_expansion_coefficients;
      out.specific_heat = base_output.specific_heat;
      out.thermal_conductivities = base_output.thermal_conductivities;
      out.compressibilities = base_output.compressibilities;
      out.entropy_derivative_pressure = base_output.entropy_derivative_pressure;
      out.entropy_derivative_temperature = base_output.entropy_derivative_temperature;
      out.reaction_terms = base_output.reaction_terms;

      // Finally, we move the additional outputs back into place:
      out.move_additional_outputs_from(base_output);

      // Diking event setup
      // 1. Find the dike location.
      double dike_position_x1 = 0;
      double dike_position_x2 = 0;
      double dike_bottom_depth = 0;

      const QGauss<dim-1> quadrature_formula_face (this->get_fe()
                                                   .base_element(this->introspection().base_elements.temperature)
                                                   .degree+1);
      FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                          this->get_fe(),
                                          quadrature_formula_face,
                                          update_values |
                                          update_quadrature_points);

      const types::boundary_id top_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      // Loop over all of the boundary cells and go to the surface cell.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          for (const unsigned int face_no : cell->face_indices())
            if (cell->face(face_no)->at_boundary())
              {
                if ( cell->face(face_no)->boundary_id() != top_boundary_id)
                  continue;

                // Focus on each quadrature point on the boundary cell's upper face if on the top boundary.
                fe_face_values.reinit (cell, face_no);
                
                for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                  {
                    // dike probability follwing the Gaussian Distribution 
                    double dike_probablity = dike_distribution(fe_face_values.quadrature_point(q));
                    double probablity_limit = A;
                    if (enable_random_noise == true)
                      probablity_limit = A * 1.05; //TODO: 1.05 is just a test number, find a better way to find the dike location.

                    if (dike_probablity >= probablity_limit)
                      {
                        // Currently, there is only one dike event in each timestep
                        // In the future, more dikes should be activated in each timestep.
                        // But how to check the dike number, or should it be fixed?
                        // Maybe define the dike_position to be a vector, such as
                        // std::vector<double> dike_position_x;
                        // dike_position_x[0 to dike_number-1] = fe_face_values.quadrature_point(q)[0];
                        dike_position_x1 = fe_face_values.quadrature_point(q)[0];
                        dike_position_x2 = dike_position_x1 + dike_width;
                        dike_bottom_depth = this->get_geometry_model().depth(fe_face_values.quadrature_point(q)) + dike_depth;
                      }
                  }
              }

      // Start to add the additional RHS terms to Stokes equations.
      MaterialModel::PrescribedPlasticDilation<dim>
      *prescribed_dilation = (this->get_parameters().enable_prescribed_dilation)
                             ? out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim> >()
                             : nullptr;
      
      double M_eff = 0;  // final effective M value
      // Specify the piece-wise linear time-dependent function of M
      if (this->get_time() < M_cutoff_time1)
        M_eff = this->get_time() * M_cutoff_value / M_cutoff_time1;
      else if (this->get_time() >= M_cutoff_time1 && this->get_time() <= M_cutoff_time2)
        M_eff = (this->get_time() - M_cutoff_time1) * (M_value - M_cutoff_value) / (M_cutoff_time2 - M_cutoff_time1) + M_cutoff_value;
      else
        M_eff = M_value;

      // Calculate the injection term in the Stokes eq.
      double injection_term =0;
      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        {
          const Point<dim> current_position = in.position[i];
          // Initial timestep will be skipped.
          // In each timestep, if the timestep is grater than the duration of
          // dike episode, we assume there is no diking?
          if (current_position[0] >= dike_position_x1 &&
              current_position[0] <= dike_position_x2 &&
              this->get_geometry_model().depth(current_position) <= dike_bottom_depth &&
              this->get_timestep() <= dike_duration && this->get_timestep_number() != 0)
            injection_term = 2 * M_eff * half_extension_rate / dike_width;

          if (prescribed_dilation != nullptr)
            {
              if (this->convert_output_to_years())
                prescribed_dilation->dilation[i] = injection_term / year_in_seconds;
              else
                prescribed_dilation->dilation[i] = injection_term;
            }

          // No plastic deformation in the dike
          const std::vector<double> &composition = in.composition[i];
          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
            {
              // Lookup the injection area              
              if (prescribed_dilation->dilation[i] != 0.0 && c == this->introspection().compositional_index_for_name("plastic_strain"))
                out.reaction_terms[i][c] = -1.0 * composition[c];
            }
        }
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Prescribed dike injection");
        {
          prm.declare_entry("Base model","simple",
                            Patterns::Selection(MaterialModel::get_valid_model_names_pattern<dim>()),
                            "The name of a material model that will be modified by an "
                            "averaging operation. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Material models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.enter_subsection("Dike injection function");
          {
            prm.declare_entry ("Dike axis line segments",
                               "",
                               Patterns::Anything(),
                               "Set the line segments that represent the dike axis. In 3d each segment is made up of "
                               "two points that represent horizontal coordinates (x,y) or (lon,lat). "
                               "The exact format for the point list describing the segments is "
                               "\"x1,y1>x2,y2;x2,y2>x3,y3;x4,y4>x5,y5\". In 2d, a segment is made up by 1 horizontal "
                               "x or longitude coordinate: \"x1;x2;x3\". Note that the segments can be connected "
                               "or isolated. The units of the coordinates are "
                               "dependent on the geometry model. In the box model they are in meters, in the "
                               "chunks they are in degrees.");
    	      prm.declare_entry ("Dike width", "200.",
      	                       Patterns::Double (0),
      	                       "The width of the magma-intrusion zone. Units: m ");
    	      prm.declare_entry ("Dike depth", "6000",
      	                       Patterns::Double (0),
      	                       "The depth of the magma-intrusion zone. Units: m ");
    	      prm.declare_entry ("Duration of the diking event", "5e3",
      	                       Patterns::Double (0),
      	                       "The duration of the activated dike. Units: year or second ");
            prm.declare_entry ("Enable random noise", "false",
                               Patterns::Bool (),
                               "Whether to add the random white noise in the Gaussian distribution");         
            prm.declare_entry ("Grid intervals for noise X or radius", "25",
                               Patterns::Integer (0),
                               "Grid intervals in X (cartesian domain) or radial (spherical) direction for the white noise "
                               "added to the initial background porosity that will then be interpolated "
                               "to the model grid. "
                               "Units: none.");
            prm.declare_entry ("Grid intervals for noise Y or longitude", "25",
                               Patterns::Integer (0),
                               "Grid intervals in Y (cartesian domain) or longitude (spherical) direction for the white noise "
                               "added to the initial background porosity that will then be interpolated "
                               "to the model grid. "
                               "Units: none.");
            prm.declare_entry ("Grid intervals for noise Z or latitude", "25",
                               Patterns::Integer (0),
                               "Grid intervals in Z (cartesian domain) or latitude (spherical) direction for the white noise "
                               "added to the initial background porosity that will then be interpolated "
                               "to the model grid. "
                               "Units: none.");
    	      prm.declare_entry ("Half extension rate", "0.0",
      	                       Patterns::Double (0),
      	                       "The velocity of half-extension. Units: m/y or m/s ");
    	      prm.declare_entry ("Melt fraction in the dike", "0.0",
      	                       Patterns::Double (0),
      	                       "M, the fraction of total extension accommodated by "
                               "the emplacement of new magmatic material. Units: none ");
    	      prm.declare_entry ("M function cutoff time1", "0.1",
      	                       Patterns::Double (0),
      	                       "Frist cutoff  time in the user-specified piece-wise time-dependent "
                               "linear function of M. Units: year or second ");
    	      prm.declare_entry ("M function cutoff time2", "0.1",
      	                       Patterns::Double (0),
      	                       "Second cutoff time in the user-specified piece-wise time-dependent "
                               "linear function of M. Units: year or second ");
    	      prm.declare_entry ("M function cutoff value", "0.0",
      	                       Patterns::Double (0),
      	                       "Cutoff value in the user-specified piece-wise time-dependent "
                               "linear function of M. Units: none ");
            prm.declare_entry ("Random number generator seed", "0",
                             Patterns::Double (0),
                             "The value of the seed used for the random number generator. "
                             "Units: none.");
            prm.declare_entry ("Maximum amplitude of Gaussian noise amplitude distribution", "1.0",
                             Patterns::Double (0),
                             "The amplitude of the Gaussian distribution of the amplitude of the dike noise. "
                             "Note that this parameter is taken to be the same for all dike segments. "
                             "Units: none.");
            prm.declare_entry ("Standard deviation of Gaussian noise amplitude distribution", "1.0",
                             Patterns::Double (0),
                             "The standard deviation of the Gaussian distribution of the amplitude of the dike noise. "
                             "Note that this parameter is taken to be the same for all dike segments. "
                             "Units: $m$ or degrees.");                               
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Prescribed dike injection");
        {
          AssertThrow( prm.get("Base model") != "prescribed dike injection",
                       ExcMessage("You may not use ''prescribed dike injection'' as the base model for itself."));

          // create the base model and initialize its SimulatorAccess base
          // class; it will get a chance to read its parameters below after we
          // leave the current section
          base_model.reset(create_material_model<dim>(prm.get("Base model")));
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(base_model.get()))
            sim->initialize_simulator (this->get_simulator());

          // Default is true, but in case we use a (ellipsoidal) chunk domain, set to false
          if (dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) == NULL)
            cartesian_domain = false;

          prm.enter_subsection("Dike injection function");
          {
            sigma                = prm.get_double ("Standard deviation of Gaussian noise amplitude distribution");
            if (!cartesian_domain)
              // convert degrees to radians for (ellipsoidal) chunks
              sigma *= numbers::PI/180.;

            A                    = prm.get_double ("Maximum amplitude of Gaussian noise amplitude distribution");
            seed                 = prm.get_double ("Random number generator seed");
            enable_random_noise  = prm.get_bool ("Enable random noise");
            dike_width           = prm.get_double ("Dike width");
            dike_depth           = prm.get_double ("Dike depth");
            dike_duration        = prm.get_double ("Duration of the diking event");
            half_extension_rate  = prm.get_double ("Half extension rate");
            M_value              = prm.get_double ("Melt fraction in the dike");
            M_cutoff_time1       = prm.get_double ("M function cutoff time1");
            M_cutoff_time2       = prm.get_double ("M function cutoff time2");
            M_cutoff_value       = prm.get_double ("M function cutoff value");
            grid_intervals[0]    = prm.get_integer ("Grid intervals for noise X or radius");
            grid_intervals[1]    = prm.get_integer ("Grid intervals for noise Y or longitude");
            if (dim == 3)
              grid_intervals[2]  = prm.get_integer ("Grid intervals for noise Z or latitude");

            // Read in the string of rift segments
            const std::string temp_all_segments = prm.get("Dike axis line segments");
            // Split the string into segment strings
            const std::vector<std::string> temp_segments = Utilities::split_string_list(temp_all_segments,';');
            // The number of segments, each consisting of a begin and an end point in 3d and one point in 2d
            const unsigned int n_temp_segments = temp_segments.size();
            point_list.resize(n_temp_segments);

            // Loop over the segments to extract the points
            for (unsigned int i_segment = 0; i_segment < n_temp_segments; i_segment++)
              {
                // In 3d a line segment consists of 2 points,
                // in 2d of only 1 (ridge axis orthogonal to x and y).
                // Also, a 3d point has 2 coordinates (x and y),
                // a 2d point only 1 (x).
                const std::vector<std::string> temp_segment = Utilities::split_string_list(temp_segments[i_segment],'>');
                if (dim == 3)
                  {
                    AssertThrow(temp_segment.size() == 2,ExcMessage ("The given coordinate '" + temp_segment[i_segment] + "' is not correct. "
                                                                     "It should only contain 2 parts: "
                                                                     "the two points of the segment, separated by a '>'."));
                  }
                else
                  {
                    // Add the point to the list of points for this segment
                    // As we're in 2d all segments correspond to 1 point consisting of 1 coordinate
                    // const double temp_point = Utilities::string_to_double(temp_segments[i_segment]);
                    AssertThrow(temp_segment.size() == 1,ExcMessage ("The given coordinate '" + temp_segment[i_segment] + "' is not correct. "
                                                                     "It should only contain 1 part: "
                                                                     "the point representing the rift axis."));
                  }

                // Loop over the 2 points of each segment
                for (unsigned int i_points = 0; i_points < dim-1; i_points++)
                  {
                    std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_segment[i_points],','));
                    if (dim == 3)
                      {
                        AssertThrow(temp_point.size() == 2,ExcMessage ("The given coordinates of segment '" + temp_segment[i_points] + "' are not correct. "
                                                                       "It should only contain 2 parts: "
                                                                       "the x and y coordinates of the segment begin/end point, separated by a ','."));
                      }
                    else
                      {
                        AssertThrow(temp_point.size() == 1,ExcMessage ("The given coordinates of segment '" + temp_segment[i_points] + "' are not correct. "
                                                                       "It should only contain 1 part: "
                                                                       "the one coordinate of the segment end point."));
                      }

                    if (!cartesian_domain)
                      {
                        // convert degrees to radians for (ellipsoidal) chunks
                        // longitude
                        temp_point[0] *= numbers::PI/180.;
                        // and convert latitude to colatitude
                        if (dim == 3)
                          temp_point[dim-2] = 0.5 * numbers::PI - temp_point[1] * numbers::PI / 180.;
                      }
                    // Add the point to the list of points for this segment
                    point_list[i_segment][i_points][0] = temp_point[0];
                    point_list[i_segment][i_points][1] = temp_point[dim-2];
                  }
              }  
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // After parsing the parameters for averaging, it is essential
      // to parse parameters related to the base model.
      base_model->parse_parameters(prm);
      this->model_dependence = base_model->get_model_dependence();

    }

    template <int dim>
    bool
    PrescribedDikeInjection<dim>::
    is_compressible () const
    {
      return base_model->is_compressible();
    }

    template <int dim>
    double
    PrescribedDikeInjection<dim>::
    reference_viscosity () const
    {
      return base_model->reference_viscosity();
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // The base model may have additional outputs, so we need to copy
      // these additional outputs.
      base_model->create_additional_named_outputs(out);

      // Because we use the force outputs in the heating model, we always
      // have to attach them, not only in the places where the RHS of the
      // Stokes system is computed.
      const unsigned int n_points = out.n_evaluation_points();

      //Stokes additional RHS
      if (this->get_parameters().enable_additional_stokes_rhs
          && out.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>> (n_points));
        }

      AssertThrow(!this->get_parameters().enable_additional_stokes_rhs
                  ||
                  out.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >()->rhs_u.size()
                  == n_points, ExcInternalError());

      // Prescribed dilation
      if (this->get_parameters().enable_prescribed_dilation
          && out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim>>() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::PrescribedPlasticDilation<dim>> (n_points));
        }

      AssertThrow(!this->get_parameters().enable_prescribed_dilation
                  ||
                  out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim> >()->dilation.size()
                  == n_points, ExcInternalError());

    }
  }
}

namespace aspect
{
  namespace HeatingModel
  {
    template <int dim>
    void
    LatentHeatDikeInjection<dim>::
    evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
              const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
              HeatingModel::HeatingModelOutputs &heating_model_outputs) const
    {
      AssertThrow(heating_model_outputs.heating_source_terms.size() == material_model_inputs.position.size(),
                  ExcMessage ("Heating outputs need to have the same number of entries as the material "
                              "model inputs."));

      const MaterialModel::PrescribedPlasticDilation<dim>
      *prescribed_dilation =
        (this->get_parameters().enable_prescribed_dilation)
        ? material_model_outputs.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim> >()
        : nullptr;

      // Add the latent heat source term corresponding to prescribed dilation
      // terms in Stokes equations to the rhs of energy conservation equation.
      for (unsigned int q=0; q<heating_model_outputs.heating_source_terms.size(); ++q)
        {
          heating_model_outputs.heating_source_terms[q] = 0.0;
          heating_model_outputs.lhs_latent_heat_terms[q] = 0.0;
          heating_model_outputs.rates_of_temperature_change[q] = 0.0;

          if (prescribed_dilation != nullptr)
            heating_model_outputs.heating_source_terms[q] = prescribed_dilation->dilation[q] * (latent_heat_of_crystallization + (temperature_of_injected_melt - material_model_inputs.temperature[q]) * material_model_outputs.densities[q] * material_model_outputs.specific_heat[q]);

        }
    }

    template <int dim>
    void
    LatentHeatDikeInjection<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Latent heat dike injection");
        {
          prm.declare_entry ("Latent heat of crystallization", "1.1e9",
                             Patterns::Double(0),
                             "The latent heat of crystallization that is released when material "
                             "is injected into the model. "
                             "Units: J/m$^3$.");
          prm.declare_entry ("Temperature of injected melt", "1473",
                             Patterns::Double(0),
                             "The temperature of the material injected into the model. "
                             "Units: K.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    LatentHeatDikeInjection<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &material_model_outputs) const
    {
      this->get_material_model().create_additional_named_outputs(material_model_outputs);
    }

    template <int dim>
    void
    LatentHeatDikeInjection<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Latent heat dike injection");
        {
          latent_heat_of_crystallization = prm.get_double ("Latent heat of crystallization");
          temperature_of_injected_melt = prm.get_double ("Temperature of injected melt");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(PrescribedDikeInjection,
                                   "prescribed dike injection",
                                   "The material model uses a ``Base model'' from which material properties are "
                                   "derived. It then adds source terms in the Stokes equations "
                                   "that describe a dike injection of melt to the model. ")
  }

  namespace HeatingModel
  {
    ASPECT_REGISTER_HEATING_MODEL(LatentHeatDikeInjection,
                                  "latent heat dike injection",
                                  "Latent heat releases due to the injection of melt into the model. "
                                  "This heating model takes the source term added to the Stokes "
                                  "equation and adds the corresponding source term to the energy "
                                  "equation. This source term includes both the effect of latent "
                                  "heat release upon crystallization and the fact that injected "
                                  "material might have a different temperature.")
  }
}