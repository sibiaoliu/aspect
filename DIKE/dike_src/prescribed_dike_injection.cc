/*
  Copyright (C) 2024 by the authors of the ASPECT code.
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

#include "prescribed_dike_injection.h"
#include <aspect/geometry_model/box.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/mesh_deformation/fastscape.h>
#include <aspect/utilities.h>
#include <aspect/parameters.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void
    PrescribedDikeInjection<dim>::initialize()
    {
      base_model->initialize();
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
      
      // If using random dike generation
      if (enable_random_dike_generation)
        {
          // Dike is randomly generated in the potential dike generation
          // zone at each timestep.
          double x_dike_location = 0.0;

          // 1. generate a random number
          // We use a fixed number as seed for random generator
          // this is important if we run the code on more than 1 processor
          std::srand(static_cast<unsigned int>((seed + 1) * this->get_timestep_number()));

          // 2. Determine the location (x_coordinate) of the generated dike
          // by appling transfer function, which is a parabolic relationship
          // between the generated random number and the dike x-coordinate.
          // i.e., rad_num =  (coefficent_a * (x_dike - (x_center_dike_generation_zone
          //                 - width_dike_generation_zone / 2)) ^2
          // rad_num = (std::rand() % 10000) / 10000.0; range is [0,1)
          // coefficent_a = 1 / (width_dike_generation_zone/2)
          double x_dike_raw = 0.5 * width_dike_generation_zone
                              * std::sqrt((std::rand() % 10000) / 10000.0)
                              + x_center_dike_generation_zone 
                              - 0.5 * width_dike_generation_zone;

          // flip a coin and distribute dikes symmetrically around the center position of
          // dike generation zone (x_center_dike_generation_zone).
          if ((std::rand() % 10000) / 10000.0 < 0.5)
            x_dike_location = x_dike_raw ;
          else
            x_dike_location = 2 * x_center_dike_generation_zone - x_dike_raw;

          // 3. Find the x-direction side boundaries of the column where the dike is located.
          // TODO: Applies to all geometry models.
          AssertThrow(Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model()),
                      ExcMessage("Currently, this function only works with the box geometry model."));

          const GeometryModel::Box<dim> &
          geometry_model
            = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());

          // Get the maximum resolution in the x direction.
          const double dx_max = geometry_model.get_extents()[0] 
                                / (geometry_model.get_repetitions()[0]
                                * std::pow(2,total_refinement_levels));

          // Here we assume that the dike width equals dx_max.
          x_dike_left_boundary = std::floor(x_dike_location / dx_max) * dx_max;
          x_dike_right_boundary = x_dike_left_boundary + dx_max;
          //std::cout << "x_dike_left_boundary: " << x_dike_left_boundary << "\n" << std::endl;
        }
      
      base_model->update();
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::evaluate(const typename Interface<dim>::MaterialModelInputs &in,
                                      typename Interface<dim>::MaterialModelOutputs &out) const
    {
      MaterialModel::PrescribedPlasticDilation<dim>
      *prescribed_dilation = (this->get_parameters().enable_prescribed_dilation)
                             ? out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim> >()
                             : nullptr;
      
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

      // The injection material will replace part of the original material based on the injection rate and diking
      // duration, i.e.,fraction of injected material to original material for the existence duration of a dike.
      double dike_injection_rate = 0.0;
      double dike_injection_fraction = 0.0;

      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        {
          // First find the location of the dike which is either randomly
          // generated or prescribed by the user.
          // Then give the dike_injection_rate to the dike points.
          if (enable_random_dike_generation)
            {
              // Note: when the dike is generated randomly, the prescribed injection
              // rate in the 'Dike injection function' should be only time dependent
              // and independent of the xyz-coordinate.
              const double point_depth = this->get_geometry_model().depth(in.position[i]);
              // Find the randomly generated dike location
              if (in.position[i][0] >= x_dike_left_boundary && in.position[i][0] <= x_dike_right_boundary
                  && in.temperature[i] <= T_bottom_random_dike && point_depth >= min_depth_random_dike
                  && this->simulator_is_past_initialization())
                dike_injection_rate = this->convert_output_to_years() 
                                      ? injection_function.value(in.position[i]) / year_in_seconds
                                      : injection_function.value(in.position[i]);
            }
          else
            dike_injection_rate = this->convert_output_to_years()
                                  ? injection_function.value(in.position[i]) / year_in_seconds
                                  : injection_function.value(in.position[i]);
          
          // Start to add the additional RHS terms of dike injection to Stokes equations.
          if (prescribed_dilation != nullptr)
            prescribed_dilation->dilation[i] = dike_injection_rate;
                    
          // Below we track the motion of injection material released from the dike.
          // User-defined or timestep-dependent injection fraction
          if (this->simulator_is_past_initialization())
            dike_injection_fraction = dike_injection_rate * this->get_timestep();

          if (dike_material_injection_fraction != 0.0)
            dike_injection_fraction = dike_material_injection_fraction;

          const std::vector<double> &composition = in.composition[i];
          // We limit the value of injection phase compostional field is [0,1] 
          double injection_phase_composition = std::max(std::min(composition[injection_phase_index],1.0),0.0); 
          if (dike_injection_rate != 0.0)
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
                      // When new dike material is injected, the other compositional fields
                      // at the dike point will be reduced in the same proportion (p_c) to
                      // ensure that the sum of all compositional fields is always 1.0.
                      // For example, in the previous step, the dike material has a compostional
                      // field with the value of c_dike_old and another compositional field
                      // with a value of c_1_old. So, c_dike_old + c_1_old = 1.0. Here we
                      // leave the background field alone, because it will be automatically
                      // populated if c_dike_old + c_1_old < 1.0.
                      // In the currest step, when adding a new dike material of amount 'c_dike_add',
                      // c_dike_new =  c_dike_old + c_dike_add. c_1_new = c_1_old * p_c.
                      // Since c_1_new + c_dike_new = 1.0, we get 
                      // p_c = (1.0 - c_dike_old - c_dike_add) / c_1_old.
                      // Then the amount of change in c_1 is:
                      // delta_c_1 = c_1_new - c_1_old = c_1_old * (p_c - 1.0)
                      // = - c_1_old * (c_dike_add / (1.0001 - c_dike_old))
                      // To avoid dividing by 0, we will use 1.0001 instead of 1.0.
                      if (composition[c] < 0.0)
                        out.reaction_terms[i][c] = -composition[c];
                      else 
                        out.reaction_terms[i][c] = -composition[c] * std::min(dike_injection_fraction / (1.0001 - injection_phase_composition), 1.0);
                    }
                }
              
              // TODO: keep the previous deformation
              // i.e., out.reaction_terms[i][plastic_strain_index] = 0; 
              // We assume no elastic and plastic deformation inside the narrow dike.
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

      // When calculating other properties such as viscosity, we need to 
      // correct for the effect of injection on the strain rate and thus
      // the deviatoric strain rate.
      MaterialModel::MaterialModelInputs<dim> in_corrected_strainrate (in);

      // fill variable out with the results form the base material model
      // The base model may have additional outputs such as frictional angle
      // in visco-plastic model, so we need to copy material properties first.
      typename Interface<dim>::MaterialModelOutputs base_output(out.n_evaluation_points(),
                                                                this->introspection().n_compositional_fields);

      // Move the additional outputs to base_output so that our models can fill
      // them if desired:
      base_output.move_additional_outputs_from(out);
      base_model->evaluate(in_corrected_strainrate, base_output);

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

      //Finally, we move the additional outputs back into place:
      out.move_additional_outputs_from(base_output);
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
          prm.declare_entry("Dike material injection fraction", "0.0", Patterns::Double(0),
                            "Amount of new injected material from the dike. Units: none.");
          prm.declare_entry("X center of the dike generation zone", "0.0", Patterns::Double(0),
                            "X_coordinate of the center of the dike generation zone. Units: m.");
          prm.declare_entry("Width of the dike generation zone", "0.0", Patterns::Double(0),
                            "Width of the dike generation zone. Units: m.");
          prm.declare_entry("Total refinement levels", "0", Patterns::Double(0),
                            "The total refinment levels in the model, which equals to the sum "
                            "of global refinement levels and adpative refinement levels. This "
                            "is used for calcuting the dike location. Units: none.");
          prm.declare_entry("Random number generator seed", "0", Patterns::Double(0),
                            "The value of the seed used for the random number generator. Units: none.");
          prm.declare_entry("Bottom temperature of randomly generated dike", "873.0", Patterns::Double(0),
                            "Temperature at the bottom of the generated dike. It usually equals to "
                            "the temperature at the intersection of the brittle-ductile transition "
                            "zone and the dike. Units: none.");
          prm.declare_entry("Minimum depth of randomly generated dike", "0.0", Patterns::Double(0),
                            "Minimum depth of the generated dike. It sets to the surface by default, "
                            "but can be set to a given depth below the surface. Units: none.");
          prm.declare_entry("Enable random dike generation", "false", Patterns::Bool (),
                            "Whether the dikes are generated randomly. If the dike is generated randomly, "
                            "the prescribed injection rate in the 'Dike injection function' should be "
                            "only time dependent and independent of the xyz-coordinate.");
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
          AssertThrow( prm.get("Base model") != "Prescribed dike injection",
                       ExcMessage("You may not use ''prescribed dike injection'' as the base model for itself."));

          // create the base model and initialize its SimulatorAccess base
          // class; it will get a chance to read its parameters below after we
          // leave the current section
          base_model = create_material_model<dim>(prm.get("Base model"));
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(base_model.get()))
            sim->initialize_simulator (this->get_simulator());

          dike_material_injection_fraction = prm.get_double ("Dike material injection fraction");
          enable_random_dike_generation = prm.get_bool("Enable random dike generation");
          x_center_dike_generation_zone = prm.get_double ("X center of the dike generation zone");
          width_dike_generation_zone = prm.get_double ("Width of the dike generation zone");
          total_refinement_levels = prm.get_double ("Total refinement levels");
          seed = prm.get_double ("Random number generator seed");          
          T_bottom_random_dike = prm.get_double ("Bottom temperature of randomly generated dike");
          min_depth_random_dike = prm.get_double ("Minimum depth of randomly generated dike");

          prm.enter_subsection("Dike injection function");
          {
            try
              {
                injection_function.parse_parameters(prm);
              }
            catch (...)
              {
                std::cerr << "ERROR: FunctionParser failed to parse\n"
                          << "\t Dike injection function\n"
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

      // If 'Free Surface' is used, please ensure the 'Surface velocity
      // projection' is vertical. If the projection is normal, which means
      // the surface mesh can deform both horizontally and vertically, this
      // may distort a surface element and give a non-positive volume fraction
      // in a quadrature point which is invalid.
      prm.enter_subsection("Mesh deformation");
      {
        // Check if "free surface" is specified in Mesh deformation boundary indicators
        std::string boundary_indicators = prm.get("Mesh deformation boundary indicators");
        std::string advection_direction = "nan"; 
        if (boundary_indicators.find("free surface") != std::string::npos)
        {
            prm.enter_subsection("Free surface");
            {
              advection_direction = prm.get("Surface velocity projection");
            }
            prm.leave_subsection();
        }
        AssertThrow(advection_direction == "vertical",
                    ExcMessage("The projection is " + advection_direction + 
                              ". However, this function currently prefers to use "
                              "vertical projection if using free surface."));
      }
      prm.leave_subsection(); 

      // // Below is only for FastScape
      // // Ensure that ASPECT_WITH_FASTSCAPE is defined when compiling ASPECT.
      // // Note that FastScape only works in the geometry model Box.
      // bool is_fastscape = this->get_mesh_deformation_handler().template has_matching_mesh_deformation_object<MeshDeformation::FastScape<dim>>();
      // else if (is_fastscape)
      // AssertThrow(is_fastscape, ExcMessage("Currently, this function only works with fastscape or free surface with vertical projection."));

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

      //Stokes additional RHS for prescribed dilation
      const unsigned int n_points = out.n_evaluation_points();
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

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(PrescribedDikeInjection,
                                   "prescribed dike injection",
                                   "The material model uses a ``Base model'' from which "
                                   "material properties are derived. It then adds source "
                                   "terms in the Stokes equations that describe a dike "
                                   "injection of melt to the model. ")
  }
}