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
#include <algorithm>
#include <vector>
#include <iostream>

#include <aspect/simulator_access.h>
#include <aspect/simulator.h>
#include <aspect/utilities.h>
#include <aspect/global.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/signaling_nan.h>

#include <aspect/heating_model/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/geometry_model/box.h>

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
        // double reference_viscosity () const override;

        void
        create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const override;

      private:
        /**
         * Parsed function that specifies the region and amount of
         * material that is injected into the model.
         */
        Functions::ParsedFunction<dim> injection_function;

        /**
         * Dike material injection ratio
         */
        double dike_material_injection_ratio;

        mutable std::vector<bool> is_dike_injection_point;

        /**
         * Pointer to the material model used as the base model.
         */
        std::unique_ptr<MaterialModel::Interface<dim> > base_model;
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
        double temperature_of_injected_material;
        /**
         * Dike material injection ratio
         */
        double dike_material_injection_ratio;
    };
  }
}

namespace aspect
{
  namespace MaterialModel
  {
    // template <int dim>
    // void
    // PrescribedDikeInjection<dim>::initialize()
    // {
    //   // dike-generation function using a radom number generator
    //   const PrescribedDikeInjection<dim> &
    //   material_model
    //     = Plugins::get_plugin_as_type<const PrescribedDikeInjection<dim>>(this->get_material_model());

    //   AssertThrow(Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model()),
    //               ExcMessage("Currently, this function only works with the box geometry model."));

    //   const GeometryModel::Box<dim> &
    //   geometry_model
    //     = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());

    //   // use a fixed number as seed for random generator
    //   // this is important if we run the code on more than 1 processor
    //   std::srand((seed + 1) * this->get_timestep_number());

    //   //half_width_dike_generation: user-define
    //   //center_dike_generation: user-define
    //   const double coefficent_a = 1 / std::pow(half_width_dike_generation, 2);
      
    //   // The x-coordinate value of the randomly generated dike in each timestep.
    //   double dike_x_coordinate = std::sqrt(rad_num / coefficent_a) + center_dike_generation - half_width_dike_generation;

    //   // // Define only x direction range
    //   // std::array<std::pair<double, double>, 1> grid_extents;
    //   // grid_extents[0].first = geometry_model.get_origin()[0];
    //   // grid_extents[0].second = geometry_model.get_extents()[0];

    //   // TableIndices<1> size_idx;
    //   // size_idx[0] = grid_intervals[0] + 1;

    //   // Table<1, double> dike_x_coordinate;
    //   // dike_x_coordinate.TableBase<1, double>::reinit(size_idx);
      
    //   // TableIndices<1> idx;
    //   // for (unsigned int i = 0; i < dike_number.size()[0]; ++i)
    //   // {
    //   //     idx[0] = i;
    //   //     // the range of random number generator is [0,1).
    //   //     double rad_num = (std::rand() % 10000) / 10000.0;
    //   //     if (rad_num > 0.5)
    //   //       dike_x_coordinate(idx) = std::sqrt(rad_num / coefficent_a) + center_dike_generation - half_width_dike_generation;
    //   //     else
    //   //       dike_x_coordinate(idx) = - (std::sqrt(rad_num / coefficent_a)  + center_dike_generation - half_width_dike_generation);
    //   // }

    //   // for (unsigned int i = 0; i < dike_number.size()[0]; ++i)
    //   // {
    //   //     idx[0] = i;
    //   //     // the range of random number generator is [0,1).
    //   //     double rad_num = (std::rand() % 10000) / 10000.0;
    //   //     temp_x_coordinate(idx) = std::sqrt(rad_num / coefficent_a) + center_dike_generation - half_width_dike_generation;
    //   // }
      
      std::srand((seed + 2) * this->get_timestep_number());
      if (((std::rand() % 10000) / 10000.0) >=0.5)
        {
          dike_x_coordinate = temp_x_coordinate;
        }
      else
        dike_x_coordinate = - temp_x_coordinate;

    //   //interpolate_dike_number = std::make_unique<Functions::InterpolatedUniformGridData<1>>(grid_extents, grid_intervals, dike_x_coordinate);

    //   base_model->initialize();
    // }

    template <int dim>
    void PrescribedDikeInjection<dim>::initialize()
    {

      Asserthrow (free surface projection -vertical)
      Asserthrow (or fastscape)
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::update()
    {
      // we get time passed as seconds (always) but may want
      // to reinterpret it in years
      if (this->convert_output_to_years())
        injection_function.set_time (this->get_time() / year_in_seconds);
      else
        injection_function.set_time (this->get_time());
      
      // use a fixed number as seed for random generator
      // this is important if we run the code on more than 1 processor
      std::srand((seed + 1) * this->get_timestep_number());

      // use the random number to create a dike_x_coordinate
      // the range of random number generator is [0,1).
      //half_width_dike_generation: user-defined w/2
      //center_dike_generation: user-defined; x0
      double rad_num = (std::rand() % 10000) / 10000.0;
      const double coefficent_a = 1 / std::pow(half_width_dike_generation, 2);
      double temp_x_coordinate = std::sqrt(rad_num / coefficent_a) + x_center_dike_generation - width_dike_generation /2;

      double dike_x_coordinate = 0.0;

      rand_num_2 = (std::rand() % 10000) / 10000.0;

      // The x-coordinate value of the randomly generated dike in each timestep.
      if (rand_num_2 >= 0.5)
        {
          dike_x_coordinate = 2 * center_dike_generation - temp_x_coordinate ;
        }
      else
        dike_x_coordinate = temp_x_coordinate;

      
      // TODO: dike_min_depth to be random
      // width_min_depth_dike_generation
      // center_min_depth_dike_generation

      base_model->update();
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::
    dike_injection_point (double dike_x_coordinate) const
    {
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model()),
                  ExcMessage("Currently, this function only works with the box geometry model."));

      const GeometryModel::Box<dim> &
      geometry_model
        = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());

      // Get the x repetitions used in the parameter file.
      const unsigned int x_repetitions = geometry_model->get_repetitions()[0];

      std::array<std::pair<double, double>, 1> grid_extents;
      grid_extents[0].first = geometry_model.get_origin()[0];
      grid_extents[0].second = geometry_model.get_extents()[0];

      // total mesh refinements?
      total_refinement_level = 5;
      highest_dx = (grid_extents[0].second - grid_extents[0].first) / (x_repetitions * std::pow(2,total_refinement_level));

      // Here we assume the dike occupies only one column of cells.
      dike_left_boundary = (dike_x_coordinate % highest_dx) * highest_dx;
      dike_right_boundary = x_left_boundary + highest_dx;

      // const Quadrature<dim> &quadrature_formula = this->introspection().quadratures.temperature;
      // const unsigned int n_q_points = quadrature_formula.size();

      // FEValues<dim> fe_values (this->get_mapping(),
      //                          this->get_fe(),
      //                          quadrature_formula,
      //                          update_values   |
      //                          update_quadrature_points |
      //                          update_JxW_values);
      // std::vector<double> temperature_values(n_q_points);

      // // reset the bool vector
      // is_dike_injection_point.resize(n_q_points, false);
      // std::fill(is_dike_injection_point.begin(), is_dike_injection_point.end(), false);

      // const double target_temperature = 873.0; // 600C isotherm
      // typename DoFHandler<dim>::active_cell_iterator target_cell;
      // //Point<dim> target_point;
      // bool found = false;

      // // Step 1: find the integration point cloest to the dike_generate point
      // // (dike_x_coordinate, dike_y_coordinate) and corresponding cell.
      // for (const auto &cell : dof_handler.active_cell_iterators())
      // {
      //     fe_values.reinit(cell);
      //     if (current_cell->at_boundary(face_no))
      //       continnue;
          
      //     for (unsigned int q = 0; q < n_q_points; ++q)
      //     {
      //         const Point<dim> &point = fe_values.quadrature_point(q);
      //         if ((temperature_values[q] - target_temperature) < && std::abs(point[0] - dike_x_coordinate) <= highest_dx)
      //         {
      //             target_cell = cell;
      //             //target_point = point;
      //             found = true;
      //             break;
      //         }
      //     }
      //     if (found)
      //         break;
      // }

      // if (!found)
      // {
      //     std::cout << "There is no dike generated at this timestep." << std::endl;
      //     return;
      // }

      // // Step 2: Target cells extend towards the surface
      // typename DoFHandler<dim>::active_cell_iterator current_cell = target_cell;
      // while (true)
      // {
      //     fe_values.reinit(current_cell);
      //     // mark all quadrature points in the target cell to the dike inection point.
      //     for (unsigned int q = 0; q < n_q_points; ++q)
      //     {
      //       //const Point<dim> &point = fe_values.quadrature_point(q);
      //       is_dike_injection_point[current_cell->vertex_dof_index(q, 0)] = true;
      //     }

      //     // find the next cell
      //     bool next_cell_found = false;
      //     for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      //     {
      //       if (current_cell->at_boundary(face_no))
      //           continue;

      //       typename DoFHandler<dim>::active_cell_iterator neighbor = current_cell->neighbor(face_no);
      //       if (neighbor->center()[0] == current_cell->center()[0] && neighbor->center()[1] > current_cell->center()[1])
      //       {
      //         current_cell = neighbor;
      //         next_cell_found = true;
      //         // mark all integration points inside this cell to be dike injection points.
      //         fe_values.reinit(current_cell);
      //         for (unsigned int q = 0; q < n_q_points; ++q)
      //         {
      //             is_dike_injection_point[current_cell->vertex_dof_index(q, 0)] = true;
      //         }

      //         break;
      //       }
      //     }
      //     if (!next_cell_found)
      //         break;
      // }
    }

    
    template <int dim>
    void
    PrescribedDikeInjection<dim>::evaluate(const typename Interface<dim>::MaterialModelInputs &in,
                                      typename Interface<dim>::MaterialModelOutputs &out) const
    {
      AssertThrow(this->introspection().compositional_name_exists("injection_phase"),
                  ExcMessage("Material model 'prescribed dike injection' only works if "
                             "there is a compositional field called 'injection_phase'. "));

      // Index for injection phase
      unsigned int injection_phase_index = this->introspection().compositional_index_for_name("injection_phase");

      // Indices for all chemical compositional fields, and not e.g., plastic strain.
      // Ensure that chemical fields are kept together, and don't have non-chemical 
      // fields between chemical fields.
      const std::vector<unsigned int> chemical_composition_indices = this->introspection().chemical_composition_field_indices();
      auto min_chemical_indices = std::min_element(chemical_composition_indices.begin(), chemical_composition_indices.end());
      auto max_chemical_indices = std::max_element(chemical_composition_indices.begin(), chemical_composition_indices.end());

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

      // Start to add the additional RHS terms to Stokes equations.
      MaterialModel::PrescribedPlasticDilation<dim>
      *prescribed_dilation = (this->get_parameters().enable_prescribed_dilation)
                             ? out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim> >()
                             : nullptr;

      // The injection material will replace part of the original material based on the dilation rate and dike's
      // duration, i.e., ratio of injected material to original material for the existence duration of a dike.
      double dike_injection_ratio = 0.0;

      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        {
          // depth = this->get_geometry_model.depth(in.position[i]);

          if (in.temperature[i]<=873 && in.position[i][0] >= x_left_boundary && in.position[i][0] <= x_right_boundary)
          // if (in.temperature[i]<=873 && in.position[i][0] >= x_left_boundary && in.position[i][0] <= x_right_boundary && depth >= min_depth_dike_generator)
            {
              // Update injection rate based on the conversion to years or not
              double injection_rate = (this->convert_output_to_years())
                                      ? injection_function.value(in.position[i]) / year_in_seconds
                                      : injection_function.value(in.position[i]);

              if (prescribed_dilation != nullptr)
                prescribed_dilation->dilation[i] = injection_rate;
            }
        }


          // User-defined or timestep-dependent injection ratio
          if (this->simulator_is_past_initialization())
            dike_injection_ratio = injection_rate * this->get_timestep();

          if (dike_material_injection_ratio != 0.0)
            dike_injection_ratio = dike_material_injection_ratio;
          
          const std::vector<double> &composition = in.composition[i];
          // We limit the value of injection phase compostional field is [0,1] 
          double injection_phase_composition = std::max(std::min(composition[injection_phase_index],1.0),0.0);          

          // for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
          //   {
          //     if (injection_function.value(in.position[i]) != 0.0)
          //       {
          //         if (c == injection_phase_index)
          //           out.reaction_terms[i][c] = -1.0 * composition[c] + 1.0;
          //         else
          //           out.reaction_terms[i][c] = -1.0 * composition[c];
          //       }
          //   }

          // Find the injection area
          if (injection_rate != 0.0)
            {
              // Loop only in chemical copositional fields
              for (unsigned int c = *min_chemical_indices; c <= *max_chemical_indices; ++c)
                {
                  if (c == injection_phase_index)
                    {
                      if (composition[c] < 0.0)
                        out.reaction_terms[i][c] = -composition[c];
                      else if ((composition[c] + dike_injection_fraction) >= 1.0)
                        out.reaction_terms[i][c] = 1.0 - composition[c];
                      else
                        out.reaction_terms[i][c] = dike_injection_fraction;
                    }
                  else
                    {
                      if (composition[c] < 0.0)
                        out.reaction_terms[i][c] = -composition[c];
                      else //To prevent division by 0, we will use 1.0001 instead of 1.0.
                      // c_old:
                      // c_dike_add: new fraction of the injection material.
                      // c_dke:
                    
                      // delta_c = - c_old * (c_dike_add / (1.0001 - c_dike))
                        out.reaction_terms[i][c] = -composition[c] * std::min(dike_injection_fraction / (1.0001 - injection_phase_composition), 1.0);
                    }
                }
              //We assume no elastic and plastic deformation inside the narrow dike.

              // test: out.reaction_terms[i][plastic_strain_index] = 0;
              // TODO: keep the previous deformation
              if (this->introspection().compositional_name_exists("plastic_strain"))
                {
                  unsigned int plastic_strain_index = this->introspection().compositional_index_for_name("plastic_strain");
                  out.reaction_terms[i][plastic_strain_index] = -composition[plastic_strain_index];
                }
              if (this->introspection().compositional_name_exists("viscous_strain"))
                {
                  unsigned int viscous_strain_index = this->introspection().compositional_index_for_name("viscous_strain");
                  out.reaction_terms[i][viscous_strain_index] = -composition[viscous_strain_index];
                }
              if (this->introspection().compositional_name_exists("total_strain"))
                {
                  unsigned int total_strain_index = this->introspection().compositional_index_for_name("total_strain");
                  out.reaction_terms[i][total_strain_index] = -composition[total_strain_index];
                }

              if (this->get_parameters().enable_elasticity)
                {
                  unsigned int index_ve_stress_xx = this->introspection().compositional_index_for_name("ve_stress_xx");
                  unsigned int index_ve_stress_yy = this->introspection().compositional_index_for_name("ve_stress_yy");
                  if (dim == 2)
                    {
                      unsigned int index_ve_stress_xy = this->introspection().compositional_index_for_name("ve_stress_xy");
                      out.reaction_terms[i][index_ve_stress_xx] = -composition[index_ve_stress_xx];
                      out.reaction_terms[i][index_ve_stress_yy] = -composition[index_ve_stress_yy];
                      out.reaction_terms[i][index_ve_stress_xy] = -composition[index_ve_stress_xy];
                    }
                  else //if (dim == 3)
                    {
                      unsigned int index_ve_stress_zz = this->introspection().compositional_index_for_name("ve_stress_zz");
                      unsigned int index_ve_stress_xy = this->introspection().compositional_index_for_name("ve_stress_xy");
                      unsigned int index_ve_stress_xz = this->introspection().compositional_index_for_name("ve_stress_xz");
                      unsigned int index_ve_stress_yz = this->introspection().compositional_index_for_name("ve_stress_yz");
                      out.reaction_terms[i][index_ve_stress_xx] = -composition[index_ve_stress_xx];
                      out.reaction_terms[i][index_ve_stress_yy] = -composition[index_ve_stress_yy];
                      out.reaction_terms[i][index_ve_stress_xy] = -composition[index_ve_stress_xy];
                      out.reaction_terms[i][index_ve_stress_zz] = -composition[index_ve_stress_zz];
                      out.reaction_terms[i][index_ve_stress_yz] = -composition[index_ve_stress_yz];
                      out.reaction_terms[i][index_ve_stress_xz] = -composition[index_ve_stress_xz];
                    }
                }
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
          prm.declare_entry("Dike material injection ratio", "0.0",
                            Patterns::Double(0),
                            "Ratio of injected material to original material for the existence "
                            "duration of a dike. Units: none.");
          prm.enter_subsection("Dike injection function");
          {
            Functions::ParsedFunction<dim>::declare_parameters(prm,1);
            prm.declare_entry("Function expression","0.0");
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
          base_model = create_material_model<dim>(prm.get("Base model"));
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(base_model.get()))
            sim->initialize_simulator (this->get_simulator());

          dike_material_injection_ratio = prm.get_double ("Dike material injection ratio");
          prm.enter_subsection("Dike injection function");
          {
            try
              {
                injection_function.parse_parameters(prm);
              }
            catch (...)
              {
                std::cerr << "ERROR: FunctionParser failed to parse\n"
                          << "\t Injection function\n"
                          << "with expression \n"
                          << "\t' " << prm.get("Function expression") << "'";
                throw;
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
      double dike_injection_ratio = 0.0;

      for (unsigned int q=0; q<heating_model_outputs.heating_source_terms.size(); ++q)
        {
          heating_model_outputs.heating_source_terms[q] = 0.0;
          heating_model_outputs.lhs_latent_heat_terms[q] = 0.0;
          heating_model_outputs.rates_of_temperature_change[q] = 0.0;
          if (prescribed_dilation != nullptr)
            {
              // User-defined or timestep-dependent injection ratio
              if (this->simulator_is_past_initialization())
                dike_injection_ratio = prescribed_dilation->dilation[q] * this->get_timestep();

              if (dike_material_injection_ratio != 0.0)
                dike_injection_ratio = dike_material_injection_ratio;

              // adding the laten heat source team
              heating_model_outputs.heating_source_terms[q] = dike_injection_ratio * prescribed_dilation->dilation[q] * (latent_heat_of_crystallization + (temperature_of_injected_material - material_model_inputs.temperature[q]) * material_model_outputs.densities[q] * material_model_outputs.specific_heat[q]);
            }
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
          prm.declare_entry ("Temperature of the injected material", "1273",
                             Patterns::Double(0),
                             "The temperature of the material injected into the model. "
                             "Units: K.");
          prm.declare_entry("Dike material injection ratio", "0.0",
                            Patterns::Double(0),
                            "Ratio of injected material to original material for the existence "
                            "duration of a dike. Units: none.");
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
          temperature_of_injected_material = prm.get_double ("Temperature of the injected material");
          dike_material_injection_ratio = prm.get_double ("Dike material injection ratio");
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
                                  "Latent heat releases due to the material injection (e.g., melt) into the model. "
                                  "This heating model takes the source term added to the Stokes "
                                  "equation and adds the corresponding source term to the energy "
                                  "equation. This source term includes both the effect of latent "
                                  "heat release upon crystallization and the fact that injected "
                                  "material might have a different temperature.")
  }
}