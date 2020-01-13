/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

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

#ifndef _aspect_initial_composition_assimilation_h
#define _aspect_initial_composition_assimilation_h

#include <aspect/initial_composition/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>
#include <aspect/compat.h>

#include "assimilation_IT.h"

#include <array>
#include <deal.II/base/function_lib.h>

namespace aspect
{
  namespace InitialComposition
  {
    using namespace dealii;

      template <int dim>
      class Assimilation : public Interface<dim>, public SimulatorAccess<dim>
      {
        public:
          /**
           * Empty Constructor.
           */
         Assimilation ();

          /**
           * Initialization function. This function is called once at the
           * beginning of the program. Checks preconditions.
           */
         virtual
         void
         initialize ();

         /**
          * A function that is called at the beginning of each time step. For
          * the current plugin, this function loads the next age files if
          * necessary and outputs a warning if the end of the set of age
          * files is reached.
          */
         virtual
         void
         update ();

         /**
           * Returns the current age component at the given position.
         */
         double
		 ascii_age (const Point<dim> &position) const;

         /**
           * Return the initial composition as a function of position. For the
           * current class, this function returns value from the text files.
           */
         double
         initial_composition (const Point<dim> &position, const unsigned int n_comp) const;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm);

        private:
          /**
            *The parameters needed for the data assimilation model
            */

          /**
             * A variable that stores the currently used data file of a series. It
             * gets updated if necessary by update().
             */
            int current_file_number;

            /**
             * Time from which on the data file with number 'First data file
             * number' is used as boundary condition. Previous to this time, 0 is
             * returned for every field. Depending on the setting of the global
             * 'Use years in output instead of seconds' flag in the input file,
             * this number is either interpreted as seconds or as years."
             */
            double first_data_file_model_time;

            /**
             * Number of the first data file to be loaded when the model time is
             * larger than 'First data file model time'.
             */
            int first_data_file_number;

            /**
              * In some cases the boundary files are not numbered in increasing but
              * in decreasing order (e.g. 'Ma BP'). If this flag is set to 'True'
              * the plugin will first load the file with the number 'First data
              * file number' and decrease the file number during the model run.
              */
             bool decreasing_file_order;

           /**
            * Directory in which the ascii age files are present.
            */
           std::string data_directory;
           /**
            * First part of filename of ascii age files. The files have to have
            * the pattern age_file_name.n.dat where n is the number of the
            * current timestep (starts from 0).
            */
           std::string data_file_name;

           /**
            * Time in model units (depends on other model inputs) between two
            * velocity files.
            */
           double data_file_time_step;

           /**
            * Weight between ge file n and n+1 while the current time is
            * between the two values t(n) and t(n+1).
            */
           double time_weight;

           /**
            * State whether we have time_dependent boundary conditions. Switched
            * off after finding no more velocity files to suppress attempts to read
            * in new files.
            */
           bool time_dependent;

           /**
            * Scale the age-dependent thermal boundary condition by a scalar factor.
            */
           double scale_factor;

           /**
            * Pointer to an object that reads and processes data we get from
            * gplates files.
            */
           std::unique_ptr<InitialTemperature::internal::AssimilationLookup<dim> > lookup;

           /**
            * Pointer to an object that reads and processes data we get from
            * gplates files. This saves the previous data time step.
            */
           std::unique_ptr<InitialTemperature::internal::AssimilationLookup<dim> > old_lookup;

           /**
                   * Handles the update of the age data in lookup. The input
                   * parameter makes sure that both age files (n and n+1) can be
                   * reloaded if the model time step is larger than the velocity file
                   * time step.
           */
           void
           update_data (const bool load_both_files);

           /**
                   * Handles settings and user notification in case the time-dependent
                   * part of the top thermal boundary condition is over.
            */
            void
            end_time_dependence ();

            /**
                   * Create a filename out of the name template.
             */
            std::string
            create_filename (const int timestep) const;

          /**
           * The maximum thickness and temperature of an oceanic plate
           * and a continental plate when time goes to infinity
           */
          double fixed_age_continent;
          double d_cc;
          double d_oc;
          double d_clm;
          double d_olm;
          double d_tzt;
          double d_tzb;
          double wzs;	//The size of weak zone

    };
  }
}


#endif
