import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def panelview(data, 
              formula=None, 
              Y=None, 
              D=None, 
              X=None, 
              index=None, 
              ignore_treat=False, 
              type="treat", 
              outcome_type="continuous", 
              treat_type=None, 
              by_group=False, 
              by_group_side=False, 
              by_timing=False, 
              theme_bw=True, 
              xlim=None, ylim=None, 
              xlab=None, ylab=None, 
              gridOff=False, legendOff=False, 
              legend_labs=None, main=None, 
              pre_post=None, id=None, 
              show_id=None, color=None, 
              axis_adjust=False, axis_lab="both", 
              axis_lab_gap=(0, 0), axis_lab_angle=None, 
              shade_post=False, 
              cex_main=15, cex_main_sub=12, 
              cex_axis=8, cex_axis_x=None, 
              cex_axis_y=None, cex_lab=12, 
              cex_legend=12, background=None, 
              style=None, by_unit=False, 
              lwd=0.2, leave_gap=False, 
              display_all=None, 
              by_cohort=False, collapse_history=None, 
              report_missing=False):
    
    # Convert to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Convert index to string or numeric depending on the content
    if pd.api.types.is_categorical_dtype(data[index[0]]):
        if pd.to_numeric(data[index[0]], errors='coerce').isna().sum() > 0:  # Contains text
            data[index[0]] = data[index[0]].astype(str)
        else:  # Contains numbers
            data[index[0]] = pd.to_numeric(data[index[0]].cat.codes, errors='coerce')
    
    # Number of unique units
    N0 = data[index[0]].nunique()

    if N0 <= 500:
        if collapse_history is None:
            collapse_history = False
        else:
            collapse_history = collapse_history

        if display_all is None:
            display_all = False
        else:
            display_all = display_all
    else:
        if collapse_history is not None:
            if display_all is None:
                display_all = False
            else:
                display_all = display_all
        else:
            if display_all is None:
                if type != "outcome":
                    collapse_history = True
                    display_all = False
                else:
                    collapse_history = False
                    display_all = False
            else:
                collapse_history = False


                
    # Check if 'leave_gap' is logical or in [0, 1]
    if not isinstance(leave_gap, bool) and leave_gap not in [0, 1]:
        raise ValueError('"leave_gap" is not a logical flag.')

    # Check if 'by_cohort' is logical or in [0, 1]
    if not isinstance(by_cohort, bool) and by_cohort not in [0, 1]:
        raise ValueError('"by_cohort" is not a logical flag.')

    # Check if 'display_all' is logical or in [0, 1]
    if not isinstance(display_all, bool) and display_all not in [0, 1]:
        raise ValueError('"display_all" is not a logical flag.')

    # Check if 'by_group_side' is logical or in [0, 1]
    if not isinstance(by_group_side, bool) and by_group_side not in [0, 1]:
        raise ValueError('"by_group_side" is not a logical flag.')

    # Check if 'by_unit' is logical or in [0, 1]
    if not isinstance(by_unit, bool) and by_unit not in [0, 1]:
        raise ValueError('"by_unit" is not a logical flag.')

    # Check if 'axis_adjust' is logical or in [0, 1]
    if not isinstance(axis_adjust, bool) and axis_adjust not in [0, 1]:
        raise ValueError('"axis_adjust" is not a logical flag.')

    # Check if 'axis_lab_angle' is numeric and within [0, 90]
    if axis_lab_angle is not None:
        if not isinstance(axis_lab_angle, (int, float)):
            raise ValueError('"axis_lab_angle" must be numeric.')
        elif axis_lab_angle < 0 or axis_lab_angle > 90:
            raise ValueError('"axis_lab_angle" needs to be in [0, 90].')

            

    # pre_post
    if pre_post is None:
        if type == "outcome":
            pre_post = True
        else:
            pre_post = False

    if not isinstance(pre_post, bool) and pre_post not in [0, 1]:
        raise ValueError('"pre_post" is not a logical flag.')

    # theme_bw
    if not isinstance(theme_bw, bool) and theme_bw not in [0, 1]:
        raise ValueError('"theme_bw" is not a logical flag.')

    # by_timing
    if not isinstance(by_timing, bool) and by_timing not in [0, 1]:
        raise ValueError('"by_timing" is not a logical flag.')

    # by_group
    if not isinstance(by_group, bool) and by_group not in [0, 1]:
        raise ValueError('"by_group" is not a logical flag.')

    # ignore_treat
    if not isinstance(ignore_treat, bool) and ignore_treat not in [0, 1]:
        raise ValueError('"ignore_treat" is not a logical flag.')

    # Ensure that 'by_group_side' implies 'by_group'
    if by_group_side:
        if not by_group:
            by_group = True

    # Warning for 'by_group' and 'by_cohort' combination
    if by_group:
        if by_cohort is not None:
            print('Warning: "by_cohort" is not allowed with "by_group = True" or "by_group_side = True". Ignored.')
            by_cohort = None  # Ignoring 'by_cohort'

    # Handle 'type = missing' and 'ignore_treat'
    if type in ["missing", "miss"]:
        if ignore_treat:
            raise ValueError('Option "type = missing" should not be combined with "ignore_treat = True".')

    # Check combination of 'by_cohort' and 'type'
    if type != "outcome" and by_cohort:
        raise ValueError('Option "by_cohort = True" should be combined with "type = \'outcome\'".')

    # Check combination of 'type = outcome' and 'collapse_history'
    if type == "outcome" and collapse_history:
        raise ValueError('Option "collapse_history = True" should not be combined with "type = \'outcome\'".')

    # Handling the formula logic
    if formula is not None:  # with formula

        # Check if the formula starts with "~"
        if formula.split("~")[0] == "":
            raise ValueError('You need to specify "Y"/"D"/"X" or provide a proper "formula".')

        # Parse the formula
        varnames = patsy.ModelDesc.from_formula(formula).lhs_termlist + patsy.ModelDesc.from_formula(formula).rhs_termlist
        varnames = [str(v) for v in varnames]  # Convert to strings

        Y = varnames[0]  # Left-hand side of the formula

        if not isinstance(Y, (int, float)):  # Y is a variable (not a number)
            # Outcome
            Y = varnames[0]

            # Treatment indicator and covariates
            if len(varnames) == 1:  # Only Y
                D = X = None
                ignore_treat = 1

                if type == "treat":  # Y ~ 1, type(treat)
                    print('"type = treat" not allowed. Plot "type = missing" instead.')
                    type = "missing"

            elif len(varnames) == 2:
                if ignore_treat == 0:
                    D = varnames[1]
                    X = None
                else:
                    D = None
                    X = varnames[1]

            else:  # len(varnames) > 2
                if ignore_treat == 0:
                    D = varnames[1]
                    X = varnames[2:]
                else:
                    D = None
                    X = varnames[1:]

        elif isinstance(Y, (int, float)):  # Y is a number
            # Outcome
            Y = None

            # Treatment indicator and covariates
            if len(varnames) == 1:  # 1 ~ D/X
                if ignore_treat == 0:  # 1 ~ D
                    D = varnames[0]
                    X = None
                else:  # 1 ~ X
                    raise ValueError("formula form not allowed.")

                if type in ["missing", "miss"]:  # 1 ~ variable, type(miss): not allowed
                    raise ValueError("formula form not allowed.")

            elif len(varnames) == 2:  # 1 ~ D + X
                if ignore_treat == 0:  # 1 ~ D + X
                    D = varnames[0]
                    X = varnames[1]
                else:  # 1 ~ X
                    raise ValueError("formula form not allowed.")

            else:  # len(varnames) > 2
                if ignore_treat == 0:
                    D = varnames[0]
                    X = varnames[1:]
                else:
                    raise ValueError("formula form not allowed.")

    else:  # No formula provided
        varnames = [Y, D, X]
        if D is None and X is None:  # Y = "Y", set type = "miss" as default
            if type == "treat":
                print('"type = treat" not allowed. Plot "type = missing" instead.')
                type = "missing"


    # Check for incorrect variable names
    for var in varnames:
        if var not in data.columns:
            raise ValueError(f'Variable "{var}" is not in the dataset.')

    # Check index specification
    if len(index) != 2 or sum([i in data.columns for i in index]) != 2:
        raise ValueError('"index" option misspecified. Try, for example, index = ["unit.id", "time"].')

    # Assign index names
    index_id = index[0]
    index_time = index[1]

    # Exclude other covariates from the dataset
    data = data[[index_id, index_time] + varnames]

    # Handle missing value report if needed
    if report_missing:
        varV = [Y, D] + X if X is not None else [Y, D]
        nv = len(varV)

        # Create a matrix for missing values
        missing_data = pd.DataFrame({
            "# Missing": data[varV].isna().sum(),
            "% Missing": round(data[varV].isna().mean() * 100, 1)
        })

        print(missing_data)
        print("\n")

    # Set leave.gap logic
    if by_cohort:
        leave_gap = 1

    # Handle missing data based on leave_gap
    if leave_gap == 0:
        data = data.dropna()
    else:
        # Create row-wise missing value counts
        data['rowmiss'] = data.isna().sum(axis=1)
        data['minrowmiss'] = data.groupby(index_id)['rowmiss'].transform('min')

        # Drop units where all periods have missing values
        data = data[data['minrowmiss'] == 0]
        data = data.drop(columns=['rowmiss', 'minrowmiss'])

    # Sort the data by index
    data = data.sort_values(by=[index_id, index_time])

    # Calculate time gap
    min_time = data[index_time].min()
    max_time = data[index_time].max()
    unique_times = data[index_time].nunique()
    time_gap = (max_time - min_time) / (unique_times - 1)
    int_time_gap = int(time_gap)

    # Calculate difference in time between consecutive observations
    data['differencetime'] = data.groupby(index_id)[index_time].diff()

    min_time_gap = data['differencetime'].min()
    max_time_gap = data['differencetime'].max()

    # Check for time gaps
    if leave_gap == 0:
        if time_gap != min_time_gap or time_gap != int_time_gap:
            print("Time is not evenly distributed (possibly due to missing data).\n")
    
    # Handle leave_gap == 1 (expand panel data)
    if leave_gap == 1:
        # Calculate time differences within each unit
        data['differencetime'] = data.groupby(index_id)[index_time].diff()
        min_time_gap = data['differencetime'].min()
        max_time_gap = data['differencetime'].max()
        divide_differencetime = max_time_gap / min_time_gap if min_time_gap != 0 else np.inf

        if time_gap != min_time_gap or int_time_gap != time_gap:
            if min_time_gap != max_time_gap and min_time_gap != 1 and divide_differencetime == int(divide_differencetime):
                # Create all combinations of 'id' and 'time' based on the minimum time gap
                g = pd.DataFrame({
                    index_id: np.repeat(data[index_id].unique(), len(np.arange(min_time, max_time + min_time_gap, min_time_gap))),
                    index_time: np.tile(np.arange(min_time, max_time + min_time_gap, min_time_gap), data[index_id].nunique())
                })
                # Merge g with the data
                data = pd.merge(g, data, how='left', on=[index_id, index_time])
            else:
                g = pd.DataFrame({
                    index_id: np.repeat(data[index_id].unique(), len(np.arange(min_time, max_time + 1))),
                    index_time: np.tile(np.arange(min_time, max_time + 1), data[index_id].nunique())
                })
                data = pd.merge(g, data, how='left', on=[index_id, index_time])

        data.drop(columns=['differencetime'], inplace=True)

    # Check for duplicated observations
    unique_label = data[index_id].astype(str) + "_" + data[index_time].astype(str)
    if unique_label.nunique() != len(data):
        raise ValueError("Unit and time variables do not uniquely identify all observations. Some may be duplicated.")

    # Limit units for gridOff
    if data[index_id].nunique() > 300 and not gridOff and type != "outcome":
        print("Number of units is more than 300, setting 'gridOff = TRUE'.")
        gridOff = True

    if display_all == False and data[index_id].nunique() > 500:
        print("Number of units is more than 500, randomly selecting 500 units for display.")
        sample_subject_ids = np.random.choice(data[index_id].unique(), 500, replace=False)
        data = data[data[index_id].isin(sample_subject_ids)]


    ##-------------------------------##
    ## Checking Other Parameters
    ##-------------------------------## 
        

    # Check the 'type' option
    if type not in ["miss", "missing", "raw", "treat", "outcome", "bivar", "bivariate"]:
        raise ValueError('"type" option misspecified.')

    # Adjust for "missing" or "miss" types
    if type in ["missing", "miss"]:
        type = "treat"
        ignore_treat = 1

    # Adjust for "raw" type
    if type == "raw":
        type = "outcome"

    # Update cex settings based on group
    if by_group or type == "outcome":
        cex_main_top = cex_main
        cex_main = cex_main_sub

    # Handle axis labels
    if cex_axis_x is None:
        cex_axis_x = cex_axis

    if cex_axis_y is None:
        cex_axis_y = cex_axis

    # Check if treatment indicator is available
    if D is None and ignore_treat == 0:
        print("No treatment indicator.")
        ignore_treat = 1

    # Check if outcomes are available for certain types
    if Y is None and type in ["outcome", "bivar", "bivariate"]:
        raise ValueError("No outcomes.")

    # Validate axis.lab option
    if axis_lab not in ["both", "unit", "time", "off"]:
        raise ValueError('"axis.lab" option misspecified. Try, for example, axis.lab = ["both", "unit", "time", "off"].')

    # Validate axis.lab.gap
    if any(np.array(axis_lab_gap) < 0):
        raise ValueError('"axis.lab.gap" should be greater than or equal to 0.')

    # Validate legend labels
    if legend_labs is not None:
        legend_labs = [str(lab) for lab in legend_labs]

    # Validate outcome.type
    if outcome_type not in ["continuous", "discrete"]:
        raise ValueError('"outcome.type" option misspecified. Try, for example, outcome_type = ["continuous", "discrete"].')

    # Handle treatment indicator checks
    d_levels = None
    d_bi = False

    # Without ignore.treat
    if ignore_treat == 0:
        if leave_gap == 0:
            if not pd.api.types.is_numeric_dtype(data[D]):
                raise ValueError("Treatment indicator should be a numeric value.")

        d_levels = sorted(data[D].unique())
        n_levels = len(d_levels)
        d_bi = d_levels == [0, 1] and n_levels == 2  # Binary treatment check

        if not d_bi and by_cohort:
            raise ValueError('Option "by.cohort = TRUE" works only with dummy treatment variable')

        if outcome_type == "discrete":
            y_levels = sorted(data[Y].unique())

        if n_levels == 1:
            print("Only one treatment level...")
            ignore_treat = 1
        else:
            if not d_bi:
                print(f"{n_levels} treatment levels.")

        # Treatment type validation
        if treat_type is not None:
            if treat_type not in ["discrete", "continuous"]:
                raise ValueError('"treat.type" must be "discrete" or "continuous".')

            if treat_type == "discrete" and n_levels >= 5:
                print("Too many treatment levels; treat as continuous.")
                treat_type = "continuous"

            if treat_type == "continuous" and n_levels <= 4:
                print("Too few treatment levels; consider setting treat_type = 'discrete'.")
        else:
            treat_type = "continuous" if n_levels > 5 else "discrete"

    else:  # ignore_treat == 1
        n_levels = 0
        treat_type = "discrete"

    # Check shade.post type
    if not isinstance(shade_post, (bool, int)):
        raise ValueError('Incorrect type for option "shade.post".')

        
    
    ## ------------------------ ##
    ## parsing data.            ##
    ## ------------------------ ##
    
    # Parsing data
    raw_id = sorted(data[index_id].unique())
    raw_time = sorted(data[index_time].unique())
    N = len(raw_id)
    TT = len(raw_time)

    # Handling input.id
    input_id = None
    if id is not None:
        if show_id is not None:
            print("Using specified id.")
        remove_id = np.setdiff1d(id, raw_id)
        if len(remove_id) != 0:
            print(f"List of units removed from dataset: {remove_id}")
            input_id = np.intersect1d(sorted(id), raw_id)
        else:
            input_id = sorted(id)
    else:
        if show_id is not None:
            if len(show_id) > N:
                raise ValueError("Length of 'show.id' should not be larger than total number of units.")
            if not isinstance(show_id[0], (int, float)):
                raise ValueError("'show.id' option misspecified. Try, for example, show_id = range(1, 100).")
            if any(show_id > N):
                raise ValueError("Some specified units are not in the data.")
            if len(np.unique(show_id)) != len(show_id):
                raise ValueError("Repeated values in 'show.id' option.")
            input_id = raw_id[show_id]
        else:
            input_id = raw_id

    # Store variable names
    data_old = data.copy()
    Yname = Y
    Dname = D

    # Check missing values
    data['rowmiss'] = data.isna().sum(axis=1)
    rowmissname = 'rowmiss'

    # Subset data if necessary
    if len(input_id) != len(raw_id):
        data = data[data[index_id].isin(input_id)]
        N = len(input_id)

    # Initialize variables
    Y, D, I, M = None, None, None, None

    # Handling leave.gap == 0 (Balanced Panel)
    if leave_gap == 0:
        if len(data) != TT * N:  # Unbalanced panel
            data[index_id] = pd.factorize(data[index_id])[0] + 1
            data[index_time] = pd.factorize(data[index_time])[0] + 1

            if Yname is not None:
                Y = np.full((TT, N), np.nan)
            I = np.zeros((TT, N))

            if ignore_treat == 0:
                D = np.zeros((TT, N))

            for i in range(len(data)):
                if Yname is not None:
                    Y[data.iloc[i][index_time] - 1, data.iloc[i][index_id] - 1] = data.iloc[i][Yname]

                if ignore_treat == 0:
                    D[data.iloc[i][index_time] - 1, data.iloc[i][index_id] - 1] = data.iloc[i][Dname]

                I[data.iloc[i][index_time] - 1, data.iloc[i][index_id] - 1] = 1

        else:  # Balanced panel
            I = np.ones((TT, N))
            if Yname is not None:
                Y = data[Yname].values.reshape((TT, N), order='F')
            if ignore_treat == 0:
                D = data[Dname].values.reshape((TT, N), order='F')

    else:  # leave.gap == 1 (Balanced panel with missing data)
        data[index_id] = pd.factorize(data[index_id])[0] + 1
        data[index_time] = pd.factorize(data[index_time])[0] + 1

        M = np.zeros((TT, N))
        for i in range(len(data)):
            M[data.iloc[i][index_time] - 1, data.iloc[i][index_id] - 1] = data.iloc[i][rowmissname]

        if Yname is not None:
            Y = np.full((TT, N), np.nan)
        I = np.zeros((TT, N))
        if ignore_treat == 0:
            D = np.zeros((TT, N))

        for i in range(len(data)):
            if Yname is not None:
                Y[data.iloc[i][index_time] - 1, data.iloc[i][index_id] - 1] = data.iloc[i][Yname]

            if ignore_treat == 0:
                D[data.iloc[i][index_time] - 1, data.iloc[i][index_id] - 1] = data.iloc[i][Dname]

            I[data.iloc[i][index_time] - 1, data.iloc[i][index_id] - 1] = 1

    # Handling collapse.history == TRUE
    if collapse_history:
        D_f = np.vstack([D, I]) if M is None else np.vstack([D, I, M])

        D_d = pd.DataFrame(D_f.T)
        ff = D_d.groupby(list(D_d.columns)).size().reset_index(name='COUNT')

        D = ff.iloc[:, :TT].T.values
        I = ff.iloc[:, TT:2 * TT].T.values

        if M is None:
            input_id = ff.iloc[:, 2 * TT].values
        else:
            M = ff.iloc[:, 2 * TT:3 * TT].T.values
            input_id = ff.iloc[:, 3 * TT].values

        N = len(input_id)

        # Sort by cohort size
        D_id = np.column_stack((np.arange(1, N + 1), input_id))
        D_id = D_id[D_id[:, 1].argsort()[::-1]]
        D_id_vec = D_id[:, 0].astype(int) - 1

        input_id = D_id[:, 1]
        D = D[:, D_id_vec]
        I = I[:, D_id_vec]
        if M is not None:
            M = M[:, D_id_vec]

    D_old = D.copy()

    # Binary treatment indicator
    if not ignore_treat and d_bi:
        if len(np.unique(D_old)) > 2:
            D[D > 1] = 1

        # Once treated, always treated
        D = np.apply_along_axis(lambda x: np.cumsum(np.nan_to_num(x)) + (x == 0).astype(int) * 0, axis=0, arr=D)
        co_total_all = TT - D.sum(axis=0)
        D = (D > 0).astype(int)

        tr_pos = np.where(D[TT - 1] == 1)[0]
        T0 = np.sum(D == 0, axis=0)[tr_pos] + 1
        T1 = np.sum(D == 1, axis=0)[tr_pos]

        T1[T1 > 1] = 0
        co_total = co_total_all[tr_pos]
        DID = len(np.unique(T0)) == 1

        if np.sum(np.abs(D_old[I == 1] - D[I == 1])) == 0:
            staggered = 1
        else:
            DID = 0
            staggered = 0
    else:
        DID = 0
        staggered = 1

    ########################################
    ## unified labels:
    ##  -200 for missing
    ##  -1 for control condition (or observed)
    ##   0 for treated pre
    ##   1 for treated post  
    ########################################
    

    obs_missing = None


    if leave_gap == 0:
        if ignore_treat == 0 and d_bi == 1:  # binary, and without ignore_treat
            con1 = type == "treat" and pre_post is True
            con2 = type == "outcome" and by_group is False

            if staggered == 1 and (con1 or con2):  # DID type data
                tr = D[TT, :] == 1  # cross-sectional: treated unit

                id_tr = np.where(tr == 1)[0]
                id_co = np.where(tr == 0)[0]

                D_tr = D[:, id_tr]
                I_tr = I[:, id_tr]
                Y_tr = Y_co = None
                if type == "outcome":
                    Y_tr = Y[:, id_tr]
                    Y_co = Y[:, id_co]

                Ntr = np.sum(tr)
                Nco = N - Ntr

                # 1. control group: -1
                obs_missing = np.full((TT, N), -1)
                # 2. add treated units
                obs_missing[:, id_tr] = D[:, id_tr]
                # 3. set missing values
                obs_missing[np.where(I == 0)] = -200  # missing -200; I==0: missings in unbalanced panel

                unit_type = np.full(N, 1)  # 1 for control; 2 for treated; 3 for reversal
                unit_type[id_tr] = 2

            else:
                unit_type = np.full(N, np.nan)  # 1 for control; 2 for treated; 3 for reversal

                for i in range(N):
                    di = D_old[:, i]
                    ii = I[:, i]

                    if len(np.unique(di[np.where(ii == 1)])) == 1:  # treated or control
                        if 0 in np.unique(di[np.where(ii == 1)]):
                            unit_type[i] = 1  # always control
                        else:
                            unit_type[i] = 2  # always treated
                    else:
                        unit_type[i] = 3  # control to treated / treated to control

                # 1. using D_old  
                obs_missing = D_old.copy()
                # 2. set controls
                obs_missing[np.where(D_old == 0)] = -1  # under control
                # 3. set missing 
                obs_missing[np.where(I == 0)] = -200  # missing

            obs_missing_treat = obs_missing.copy()
            if len(np.unique(D_old)) > 2:
                obs_missing[np.where(obs_missing > 1)] = 1

        else:  # either not binary (>2 treatment levels) or ignore_treat == 1
            if n_levels > 2 and type == "treat":  # >2 treatment levels
                obs_missing = D.copy()
                obs_missing[np.where(I == 0)] = np.nan
            else:
                obs_missing = np.full((TT, N), -1)
                obs_missing[np.where(I == 0)] = -200  # missing
                ignore_treat = 1

    elif leave_gap == 1:
        if ignore_treat == 0 and d_bi == 1:  # binary, and without ignore_treat
            con1 = type == "treat" and pre_post is True
            con2 = type == "outcome" and by_group is False

            if staggered == 1 and (con1 or con2):  # DID type data
                tr = D[TT, :] == 1  # cross-sectional: treated unit

                id_tr = np.where(tr == 1)[0]
                id_co = np.where(tr == 0)[0]

                D_tr = D[:, id_tr]
                I_tr = I[:, id_tr]
                Y_tr = Y_co = None

                if type == "outcome":
                    Y_tr = Y[:, id_tr]
                    Y_co = Y[:, id_co]

                Ntr = np.sum(tr)
                Nco = N - Ntr

                # 1. control group: -1
                obs_missing = np.full((TT, N), -1)
                # 2. add treated units
                obs_missing[:, id_tr] = D[:, id_tr]
                # 3. set missing values
                obs_missing[np.where(I == 0)] = -200  # missing -200
                obs_missing[np.where(M != 0)] = -200

                unit_type = np.full(N, 1)  # 1 for control; 2 for treated; 3 for reversal
                unit_type[id_tr] = 2

            else:
                unit_type = np.full(N, np.nan)  # 1 for control; 2 for treated; 3 for reversal

                for i in range(N):
                    di = D_old[:, i]
                    ii = I[:, i]  # I: observed or missing

                    if len(np.unique(di[np.where(ii == 1)])) == 1:  # treated or control
                        if 0 in np.unique(di[np.where(ii == 1)]):
                            unit_type[i] = 1  # control
                        else:
                            unit_type[i] = 2  # treated
                    elif len(np.unique(di[np.where(ii == 1)])) == 2 and np.nan in np.unique(di[np.where(ii == 1)]):
                        if 0 in np.unique(di[np.where(ii == 1)]):
                            unit_type[i] = 1  # control
                        else:
                            unit_type[i] = 2  # treated
                    else:
                        unit_type[i] = 3  # control to treated / treated to control / NA 0 1 / NA 1 0

                # 1. using D_old  
                obs_missing = D_old.copy()
                # 2. set controls
                obs_missing[np.where(D_old == 0)] = -1  # under control
                # 3. set missing 
                obs_missing[np.where(I == 0)] = -200  # missing
                obs_missing[np.where(M != 0)] = -200

            obs_missing_treat = obs_missing.copy()
            if len(np.unique(D_old)) > 2:
                obs_missing[np.where(obs_missing > 1)] = 1

        else:  # either not binary (>2 treatment levels) or ignore_treat == 1
            if n_levels > 2 and type == "treat":  # >2 treatment levels
                obs_missing = D.copy()
                obs_missing[np.where(I == 0)] = -200
                obs_missing[np.where(M != 0)] = -200
            else:
                obs_missing = np.full((TT, N), -1)
                obs_missing[np.where(I == 0)] = -200
                obs_missing[np.where(M != 0)] = -200
                ignore_treat = 1

    # Setting column and row names
    obs_missing = pd.DataFrame(obs_missing, columns=input_id, index=raw_time)

    # Further processing for unique treatment histories (if needed)
    if collapse_history:
        obs_missing_d = pd.DataFrame(obs_missing.T)
        ff = obs_missing_d.groupby(list(obs_missing_d.columns)).size().reset_index(name="COUNT")

        obs_missing = ff.iloc[:, :-1].T.values
        input_id = ff.iloc[:, -1].values
        obs_missing = pd.DataFrame(obs_missing, columns=input_id, index=raw_time)
        N = len(input_id)

    # Setting time and id for the final output
    time = raw_time
    id = input_id


        
    ## ------------------------------------- ##
    ##          part 2: plot
    ## ------------------------------------- ##
    
        

        

    # Initialize variables
    outcome = None
    treatment = None
    labels1 = None
    labels2 = None
    labels3 = None

    # Check xlim
    if xlim is not None:
        if not isinstance(xlim, (list, tuple)) or not all(isinstance(x, (int, float)) for x in xlim):
            raise ValueError('"xlim" must be a list or tuple of two numeric values.')
        if len(xlim) != 2:
            raise ValueError('"xlim" must be of length 2.')

    # Check ylim if type is not "bivar" or "bivariate"
    if type not in ["bivar", "bivariate"]:
        if ylim is not None:
            if not isinstance(ylim, (list, tuple)) or not all(isinstance(y, (int, float)) for y in ylim):
                raise ValueError('"ylim" must be a list or tuple of two numeric values.')
            if len(ylim) != 2:
                raise ValueError('"ylim" must be of length 2.')

    # Check xlab
    if xlab is not None:
        if not isinstance(xlab, str):
            raise ValueError('"xlab" must be a string.')
        xlab = xlab

    # Check ylab
    if ylab is not None:
        if not isinstance(ylab, str):
            raise ValueError('"ylab" must be a string.')
        ylab = ylab

    # Check legendOff
    if not isinstance(legendOff, (bool, int)):
        raise ValueError('"legendOff" must be a logical flag or numeric.')

    # Check gridOff
    if not isinstance(gridOff, (bool, int)):
        raise ValueError('"gridOff" must be a logical flag or numeric.')

    # Check main
    if main is not None:
        if not isinstance(main, str):
            raise ValueError('"main" must be a string.')
        main = main

    # Set axis.lab.angle and adjust axis labels
    if axis_lab_angle is not None:
        angle = axis_lab_angle
        x_v, x_h = 1, 1
    else:
        if axis_adjust:
            angle = 45
            x_v, x_h = 1, 1
        else:
            angle = 0
            x_v = 0
            x_h = 0.5 if type == "treat" else 0

    # Handle time type
    if not isinstance(time[0], (int, float)):
        time = list(range(1, TT + 1))

    # Periods to show
    if xlim is not None:
        show = [i for i, t in enumerate(time) if xlim[0] <= t <= xlim[1]]
    else:
        show = list(range(len(time)))

    nT = len(show)
    time_label = [raw_time[i] for i in show]
    T_b = list(range(1, len(show) + 1))

    # Handle labels
    N_b = list(range(1, N + 1))
    if type == "treat":
        if axis_lab == "both":
            if len(axis_lab_gap) == 2:
                x_gap, y_gap = axis_lab_gap
            else:
                x_gap = y_gap = axis_lab_gap[0]
        else:
            x_gap = y_gap = axis_lab_gap[0]

        if y_gap != 0:
            N_b = list(range(N, 0, -(y_gap + 1)))
    else:
        x_gap = axis_lab_gap[0]

    if x_gap != 0:
        T_b = list(range(1, len(show) + 1, x_gap + 1))

    # Legend on/off
    legend_pos = "none" if legendOff == 1 else "bottom"    
    
    
    ###########################
    ## Outcome Plot
    ###########################

    if type == "outcome":


        # Axes labels
        if xlab is None:
            xlab = index[1]  # Assuming index is a list, Python is 0-indexed
        elif xlab == "":
            xlab = None

        if ylab is None:
            ylab = Yname  # Assuming Yname is a predefined variable
        elif ylab == "":
            ylab = None

        # Plot color setting
        raw_color = None

        # Color setting
        if color is None:
            if not ignore_treat:
                if outcome_type == "continuous":
                    raw_color = ["#5e5e5e50", "#FC8D62", "red"]
                else:
                    raw_color = ["#5e5e5e60", "#FC8D62", "red"]
                
                if type == "outcome" and (staggered == 0 or by_group or not pre_post):  # Two conditions only
                    raw_color = [raw_color[0], raw_color[2]]
            else:  # Ignore treat
                raw_color = "#5e5e5e50"
        else:  # Color is specified
            if not ignore_treat:
                if staggered == 0 or not pre_post:  # With reversals or two groups only
                    if len(color) != 2:
                        raise ValueError("Length of 'color' should be equal to 2.")
                    else:
                        print('Specified colors are in the order of "under treatment", "under control".')
                        raw_color = [color[1], color[0]]
                else:
                    if not by_group and len(color) != 3:
                        raise ValueError('Length of "color" should be equal to 3.')
                    elif by_group and len(color) != 2:
                        raise ValueError('Length of "color" should be equal to 2.')
                    elif not by_group and len(color) == 3:
                        print('Specified colors in the order of "treated (pre)", "treated (post)", "control".')
                        raw_color = [color[2], color[0], color[1]]
                    else:
                        print('Specified colors in the order of "under treatment", "under control".')
                        raw_color = [color[1], color[0]]
            else:
                if len(color) != 1:
                    raise ValueError('Length of "color" should be equal to 1.')


        #####################
        ## Prepare to plot
        #####################


        # Do not show treatment status
        if ignore_treat == True:

            # Create the data frame
            data = pd.DataFrame({
                'time': np.tile(time[show], N),  # Replicate `time[show]` N times
                'outcome': Y[show, :].flatten(),  # Flatten the Y matrix after indexing
                'type': np.repeat("co", N * nT),  # Repeating "co" for all observations
                'id': np.repeat(np.arange(1, N + 1), nT)  # Replicate IDs
            })

            # Handle discrete outcome
            if outcome_type == "discrete":
                data.dropna(inplace=True)
                data['outcome'] = data['outcome'].astype('category')

            # Begin plotting
            fig, ax = plt.subplots()
            if outcome_type == "continuous":
                sns.lineplot(data=data, x='time', y='outcome', hue='type', size='type', style='type', ax=ax, 
                            palette=raw_color, linewidth=0.5)

                # Legend customization
                set_limits = "co"
                set_colors = raw_color
                set_linetypes = "solid"
                set_linewidth = 0.5

                if legend_labs:
                    if len(legend_labs) != 1:
                        print("Warning: Incorrect number of labels in the legend. Using default.")
                        set_labels = ["Observed"]
                    else:
                        set_labels = legend_labs
                else:
                    set_labels = ["Observed"]
                
                ax.legend(title=None, ncol=1, labels=set_labels)
            else:
                # For categorical data
                sns.scatterplot(data=data, x='time', y='outcome', hue='type', style='type', ax=ax, jitter=True)

                # Legend customization
                set_limits = "co"
                set_colors = raw_color
                set_shapes = 1

                if legend_labs:
                    if len(legend_labs) != 1:
                        print("Warning: Incorrect number of labels in the legend. Using default.")
                        set_labels = ["Observed"]
                    else:
                        set_labels = legend_labs
                else:
                    set_labels = ["Observed"]
                
                ax.legend(title=None, ncol=1, labels=set_labels)

            # Customize axis labels
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)

            # Apply additional theming
            if theme_bw:
                sns.set_theme(style="whitegrid")

            # Adjust title
            if main is None:
                ax.set_title("Raw Data", fontsize=cex_main, fontweight='bold')
            elif main != "":
                ax.set_title(main, fontsize=cex_main, fontweight='bold')

            # Set x-axis label rotation
            plt.xticks(rotation=angle, ha='right' if x_h == 1 else 'center')

            # Handle x-axis labels if not numeric
            if not isinstance(time.label, (int, float)):
                ax.set_xticks(time[show[T_b]])
                ax.set_xticklabels(time_label[T_b])

            # Set ylim if provided
            if ylim is not None:
                ax.set_ylim(ylim)

            # Display plot
            plt.show()
        elif ((ignore_treat == False) and ( by_group == False)):  # Mixed units
            
            # time-line
            if outcome_type == "continuous":  # continuous outcome
                
                if ((staggered == 0) or (not by_cohort and not pre_post)):
                    
                    # Step 1: Prepare D.plot by replacing 0 values with NaN based on D_old and I
                    D_plot = D_old.copy()
                    D_plot[D_plot == 0] = np.nan
                    D_plot[I == 0] = np.nan

                    # Step 2: Prepare Y.trt and time.trt.show
                    Y_trt = Y * D_plot  # Element-wise multiplication to create Y.trt
                    Y_trt_show = Y_trt[show, :]  # Subset for showing
                    time_trt_show = time[show]  # Subset time for showing

                    # Step 3: Identify treatment times and IDs (ut.time and ut.id)
                    ut_time = []
                    ut_id = []
                    for i in range(N):
                        if np.sum(np.isnan(Y_trt_show[:, i])) != nT:  # Check if all values are not NaN
                            ut_id.extend([i + 1] * (nT - np.sum(np.isnan(Y_trt_show[:, i]))))  # Add i+1 (ID) to ut_id
                            ut_time.extend(time_trt_show[np.where(~np.isnan(Y_trt_show[:, i]))[0]])  # Add valid times to ut_time

                    # Step 4: Calculate T1_0, T1_1, N_T1_1, N_T1_0
                    T1_0 = np.where(T1 == 0)[0]
                    T1_1 = np.where(T1 == 1)[0]
                    N_T1_1 = len(T1_1)
                    N_T1_0 = N * nT + len(ut_id) - N_T1_1

                    # Step 5: Create the data frame
                    data = pd.DataFrame({
                        'time': np.concatenate([np.tile(time[show], N), ut_time]),
                        'outcome': np.concatenate([Y[show, :].flatten(), Y_trt_show[~np.isnan(Y_trt_show)].flatten()]),
                        'type': np.concatenate([np.repeat("co", N * nT), np.repeat("tr", len(ut_id))]),
                        'last_dot': np.concatenate([np.repeat("0", N_T1_0), np.repeat("1", N_T1_1)]),
                        'id': np.concatenate([np.repeat(np.arange(1, N + 1), nT), ut_id])
                    })

                    # Step 6: Calculate idtimes and update last_dot
                    data['idtimes'] = [sum(data['id'][:x] == data['id'][x-1]) for x in range(1, len(data['id']) + 1)]
                    data['idtimes'] = data.groupby('id')['idtimes'].transform(np.max)
                    data['last_dot'] = 0
                    data.loc[data['idtimes'] == 1, 'last_dot'] = 1

                    # Step 7: Set legend labels, colors, and other plot properties
                    set_limits = ["co", "tr"]
                    set_colors = raw_color
                    set_linetypes = ["solid", "solid"]
                    set_linewidth = [0.5, 0.5]

                    if legend_labs:
                        if len(legend_labs) != 2:
                            print("Warning: Incorrect number of labels in the legend. Using default.")
                            set_labels = ["Control", "Treated"]
                        else:
                            set_labels = legend_labs
                    else:
                        set_labels = ["Control", "Treated"]

                    labels_ncol = 2

                else:

                    # Calculate the post-treatment time points for treated units
                    time_bf = np.unique(time[T0])
                    pst = D_tr.copy()

                    for i in range(Ntr):
                        pst[T0[i], i] = 1

                    time_pst = pst[:, :].flatten() * time[:, None].flatten()
                    time_pst = time_pst[time_pst != 0]
                    Y_tr_pst = Y[:, :Ntr].flatten()[pst[:, :].flatten() == 1]

                    id_tr_pst = np.tile(np.arange(1, Ntr + 1), (nT, 1)).flatten()[pst[:, :].flatten() == 1]
                    T1_0 = T1[T1 == 0]
                    T1_1 = T1[T1 == 1]
                    N_T1_1 = len(T1_1)
                    N_T1_0 = Nco * nT + Ntr * nT + len(Y_tr_pst) - N_T1_1

                    # Combine data
                    data_dict = {
                        "time": np.concatenate([np.tile(time, N), time_pst]),
                        "outcome": np.concatenate([Y[:, :N].flatten(), Y_tr_pst]),
                        "type": np.concatenate([np.repeat("tr", Ntr * nT), np.repeat("co", Nco * nT), np.repeat("tr.pst", len(Y_tr_pst))]),
                        "last_dot": np.concatenate([np.repeat("0", N_T1_0), np.repeat("1", N_T1_1)]),
                        "id": np.concatenate([np.repeat(np.arange(1, N + 1), nT), id_tr_pst])
                    }

                    df = pd.DataFrame(data_dict)

                    # Calculate idtimes and last_dot
                    df['idtimes'] = df.groupby('id').cumcount() + 1
                    df['last_dot'] = np.where(df['idtimes'] == df.groupby('id')['idtimes'].transform('max'), 1, 0)

                    # Legend settings
                    set_limits = ["co", "tr", "tr.pst"]
                    set_colors = ['#5e5e5e50', '#FC8D62', 'red']  # Corresponding to "co", "tr", "tr.pst"
                    set_linetypes = ['solid', 'solid', 'solid']
                    set_linewidths = [0.5, 0.5, 0.5]

                    # Define the legend labels
                    if legend_labs is not None:
                        if len(legend_labs) != 3:
                            print("Warning: Incorrect number of labels in the legend. Using default.")
                            set_labels = ["Controls", "Treated (Pre)", "Treated (Post)"]
                        else:
                            set_labels = legend_labs
                    else:
                        set_labels = ["Controls", "Treated (Pre)", "Treated (Post)"]

                    if by_cohort == True:


                        # Expand to balanced panel
                        ref = pd.MultiIndex.from_product([data_old[index_id].unique(), data_old[index_time].unique()], names=[index_id, index_time])
                        ref_df = pd.DataFrame(index=ref).reset_index()

                        # Merge with the original data
                        data_old = pd.merge(ref_df, data_old, on=[index_id, index_time], how='left')

                        # Rename the column
                        data_old.columns.values[3] = 'treatment'

                        # Fill missing values
                        def fill_treatment(x):
                            filled = pd.Series(x).interpolate(method='pad', limit_direction='forward').interpolate(method='bfill')
                            return filled.fillna(method='bfill').fillna(method='pad')

                        data_old['treatment'] = data_old.groupby(index_id)['treatment'].transform(fill_treatment)

                        # Treatment history
                        data_old['treatment_history'] = data_old.groupby(index_id)['treatment'].transform(lambda x: '_'.join(map(str, x)))

                        # Check the number of unique treatment histories
                        num_unique_histories = len(data_old['treatment_history'].unique())
                        print(f"Number of unique treatment histories: {num_unique_histories}")

                        # If number of unique treatment histories exceeds 20, stop the process
                        if num_unique_histories > 20:
                            raise ValueError('"by.cohort = TRUE" ignored the number of unique treatment history is more than 20.')
                        else:

                            # Assuming data.old is a pandas DataFrame
                            # Calculate mean outcome per treatment_history and time group
                            data_old = data_old.copy()
                            data_old['outcomehistorymean'] = data_old.groupby(['treatment_history', 'time'])[data_old.columns[2]].transform(lambda x: x.mean(skipna=True))

                            # Select and rename columns
                            data_old = data_old[['time', 'treatment', 'treatment_history', 'outcomehistorymean']]
                            data_old.rename(columns={'outcomehistorymean': 'outcome', 'treatment_history': 'id'}, inplace=True)

                            # Remove duplicate rows (if needed, uncomment)
                            # data_old = data_old.drop_duplicates()

                            # Calculate the number of unique ids (cohort size)
                            N_cohort = len(data_old['id'].unique())

                            # Convert id and time columns to numeric values
                            data_old['id'] = pd.factorize(data_old['id'])[0] + 1
                            data_old['time'] = pd.factorize(data_old['time'])[0] + 1

                            # Define the time periods (TT) and cohort size (N_cohort)
                            TT = data_old['time'].max()

                            # Initialize Y and D matrices
                            Y = np.full((TT, N_cohort), np.nan)
                            D = np.zeros((TT, N_cohort))

                            # Fill Y and D matrices
                            for _, row in data_old.iterrows():
                                Y[int(row['time']) - 1, int(row['id']) - 1] = row['outcome']
                                D[int(row['time']) - 1, int(row['id']) - 1] = row['treatment']

                            # Identify treated units and cohort sizes
                            tr = D[-1, :] == 1
                            Ntr = np.sum(tr)
                            Nco = N_cohort - Ntr

                            # Split Y matrix into treated (Y.tr) and control (Y.co)
                            Y_tr = Y[:, tr == 1]
                            Y_co = Y[:, tr == 0]

                            # Get indices of treated units
                            tr_pos = np.where(D[-1, :] == 1)[0]

                            # Count the number of periods exposed to treatment for treated units
                            T1 = np.sum(D[:, tr_pos] == 1, axis=0)
                            T1[T1 > 1] = 0  # Indicate the last dot of treatment status change

                            # Treatment matrix for treated units
                            pst = D[:, tr == 1]

                            # Show time period (You need to define 'show' based on your data context)
                            # Assuming 'show' is a list or numpy array with the indices of time periods to consider
                            show = np.arange(TT)  # Placeholder, adjust as needed

                            time_pst = (pst[show, :] * np.array(show).reshape(-1, 1)).flatten()
                            time_pst = time_pst[pst[show, :].flatten() == 1]

                            # Get corresponding outcomes for treated units
                            Y_tr_pst = Y_tr[show, :].flatten()
                            Y_tr_pst = Y_tr_pst[pst[show, :].flatten() == 1]

                            # Create id matrix for treated units
                            id_tr_pst = np.repeat(np.arange(1, Ntr + 1), TT).reshape(TT, Ntr)[show, :].flatten()
                            id_tr_pst = id_tr_pst[pst[show, :].flatten() == 1]

                            # Identify T1 equals 0 or 1 cases
                            T1_0 = T1[T1 == 0]
                            T1_1 = T1[T1 == 1]

                            # Count the number of cases with T1 equal to 1 and 0
                            N_T1_1 = len(T1_1)
                            N_T1_0 = Nco * TT + Ntr * TT + len(Y_tr_pst) - N_T1_1
                        
                            if pre_post == True:
                                # Create the dataframe for pre-post analysis
                                data = pd.DataFrame({
                                    'time': np.concatenate([np.repeat(time[show], N_cohort), time_pst]),
                                    'outcome': np.concatenate([Y_tr[show, :].flatten(), Y_co[show, :].flatten(), Y_tr_pst]),
                                    'type': np.concatenate([np.repeat('tr', Ntr * nT), np.repeat('co', Nco * nT), np.repeat('tr.pst', len(Y_tr_pst))]),
                                    'id': np.concatenate([np.repeat(np.arange(1, N_cohort + 1), nT), id_tr_pst + N0])
                                })
                            else:
                                # Initialize the 'type' vector for control ('co') units and assign 'tr' to treated units
                                tr_vec = np.repeat('co', nT * N_cohort)
                                tr_vec[np.where(pst[show, :].flatten() == 1)] = 'tr'
                                
                                # Create the dataframe for non-pre-post analysis
                                data = pd.DataFrame({
                                    'time': np.repeat(time[show], N_cohort),
                                    'outcome': np.concatenate([Y_tr[show, :].flatten(), Y_co[show, :].flatten()]),
                                    'type': tr_vec,
                                    'id': np.repeat(np.arange(1, N_cohort + 1), nT)
                                })

                                # Legend settings
                                set_limits = ['co', 'tr']
                                set_colors = raw_color[:2]  # Assuming 'raw_color' is defined
                                set_linetypes = ['solid', 'solid']
                                set_linewidth = [0.5, 0.5]

                                # Check legend labels
                                if legend_labs is not None:
                                    if len(legend_labs) != 2:
                                        print("Warning: Incorrect number of labels in the legend. Using default.\n")
                                        set_labels = ['Controls', 'Treated']
                                    else:
                                        set_labels = legend_labs
                                else:
                                    set_labels = ['Controls', 'Treated']

                                labels_ncol = 2

                            # Add 'idtimes' column: count occurrences of 'id' up to each row
                            data['idtimes'] = data.groupby('id').cumcount() + 1

                            # Add 'idtimes' maximum per group
                            data['idtimes'] = data.groupby('id')['idtimes'].transform('max')

                            # Initialize the 'last_dot' column
                            data['last_dot'] = 0

                            # Set 'last_dot' to 1 where 'idtimes' equals 1
                            data.loc[data['idtimes'] == 1, 'last_dot'] = 1

                # Create a figure and axis and store it as an object p
                fig, ax = plt.subplots()

                # Assign the figure and axes to a variable, similar to storing the plot in R
                p = {
                    'figure': fig,
                    'axis': ax
                }

                # Set axis labels
                p['axis'].set_xlabel(xlab)
                p['axis'].set_ylabel(ylab)

                # Apply theme settings (similar to ggplot's theme_bw())
                if theme_bw:
                    sns.set_style("whitegrid")

                # Customize title and axis text
                p['axis'].set_title("", fontsize=cex_main, fontweight='bold', pad=10)
                plt.xticks(rotation=angle, ha='center')

                # Plot DID and Ntr conditions
                if DID and Ntr >= 1:
                    if 'time_bf' in globals() or 'time_bf' in locals():
                        if time_bf >= min(show) and time_bf <= max(show):
                            p['axis'].axvline(x=time_bf, color='white', linewidth=2)
                            if shade_post:
                                p['axis'].axvspan(time_bf, max(show), color='gray', alpha=0.3)

                # Main line plot
                sns.lineplot(data=data, x='time', y='outcome', hue='type', size='type', style='type', units='id', estimator=None, ax=p['axis'])

                # Add points for the last dot in the data
                data1 = data[data['last_dot'] == 1]
                p['axis'].scatter(data1['time'], data1['outcome'], color=raw_color[2], s=10)

                # Customize the color, linetype, and size based on 'type'
                custom_palette = dict(zip(set_limits, set_colors))
                sns.lineplot(data=data, x='time', y='outcome', hue='type', size='type', style='type', ax=p['axis'], palette=custom_palette)

                # Customizing legend
                handles, labels = p['axis'].get_legend_handles_labels()
                p['axis'].legend(handles=handles[:len(set_labels)], labels=set_labels, ncol=labels_ncol, loc=legend_pos, frameon=False)

                # Store this plot as an object 'p'
                p['figure'].tight_layout()

            else:

                

                        
                    
                






        
        

        

    ######################
    ## Mixed Units Plot
    ######################

    # Continuous outcome handling
    if outcome_type == "continuous":
        if staggered == 0 or (by_cohort == False and pre_post == False):  # With reversals
            D_plot = D_old.copy()
            D_plot[D_plot == 0] = np.nan
            D_plot[I == 0] = np.nan

            Y_trt = Y * D_plot
            Y_trt_show = Y_trt[show, :]
            time_trt_show = time[show]

            # Prepare treated and control data
            ut_time, ut_id = [], []
            for i in range(N):
                if not np.all(np.isnan(Y_trt_show[:, i])):
                    ut_id.extend([i] * (nT - np.sum(np.isnan(Y_trt_show[:, i]))))
                    ut_time.extend(time_trt_show[~np.isnan(Y_trt_show[:, i])])

            T1_0 = np.sum(T1 == 0)
            T1_1 = np.sum(T1 == 1)
            N_T1_1 = T1_1
            N_T1_0 = N * nT + len(ut_id) - N_T1_1

            # Combine data for plotting
            data = pd.DataFrame({
                "time": np.concatenate([np.tile(time[show], N), ut_time]),
                "outcome": np.concatenate([Y[show, :].flatten(), Y_trt_show[~np.isnan(Y_trt_show)]]),
                "type": np.concatenate([["co"] * (N * nT), ["tr"] * len(ut_id)]),
                "last_dot": np.concatenate([["0"] * N_T1_0, ["1"] * N_T1_1]),
                "id": np.concatenate([np.repeat(range(1, N + 1), nT), ut_id])
            })

            # Generate idtimes and last_dot
            data["idtimes"] = data.groupby("id").cumcount() + 1
            data["last_dot"] = np.where(data.groupby("id")["idtimes"].transform('max') == 1, 1, 0)

            # Set legend labels and properties
            set_limits = ["co", "tr"]
            set_colors = raw_color
            set_linetypes = ["solid", "solid"]
            set_linewidth = [0.5, 0.5]
            set_labels = ["Control", "Treated"] if legend_labs is None or len(legend_labs) != 2 else legend_labs
            labels_ncol = 2

        else:  # Staggered
            # Handling staggered cases and post-treatment periods
            time_bf = time[np.unique(T0)]
            pst = D_tr.copy()

            for i in range(Ntr):
                pst[T0[i], i] = 1

            time_pst = pst[show, :] * time[show][:, None]
            time_pst = time_pst.flatten()
            time_pst = time_pst[time_pst != 0]

            Y_tr_pst = Y_tr[show, :].flatten()
            Y_tr_pst = Y_tr_pst[time_pst != 0]

            id_tr_pst = np.tile(np.arange(1, Ntr + 1), (TT, 1))[show, :].flatten()
            id_tr_pst = id_tr_pst[time_pst != 0]

            T1_0 = np.sum(T1 == 0)
            T1_1 = np.sum(T1 == 1)
            N_T1_1 = T1_1
            N_T1_0 = Nco * nT + Ntr * nT + len(Y_tr_pst) - N_T1_1

            # Combine data for plotting
            data = pd.DataFrame({
                "time": np.concatenate([np.tile(time[show], N), time_pst]),
                "outcome": np.concatenate([Y_tr[show, :].flatten(), Y_co[show, :].flatten(), Y_tr_pst]),
                "type": np.concatenate([["tr"] * (Ntr * nT), ["co"] * (Nco * nT), ["tr.pst"] * len(Y_tr_pst)]),
                "last_dot": np.concatenate([["0"] * N_T1_0, ["1"] * N_T1_1]),
                "id": np.concatenate([np.repeat(range(1, N + 1), nT), id_tr_pst + N0])
            })

            # Generate idtimes and last_dot
            data["idtimes"] = data.groupby("id").cumcount() + 1
            data["last_dot"] = np.where(data.groupby("id")["idtimes"].transform('max') == 1, 1, 0)

            # Set legend labels and properties
            set_limits = ["co", "tr", "tr.pst"]
            set_colors = raw_color
            set_linetypes = ["solid", "solid", "solid"]
            set_linewidth = [0.5, 0.5, 0.5]
            set_labels = ["Controls", "Treated (Pre)", "Treated (Post)"] if legend_labs is None or len(legend_labs) != 3 else legend_labs
            labels_ncol = 3

        # Plot theme
        fig, ax = plt.subplots()
        sns.set(style="whitegrid" if theme_bw else "darkgrid")

        # Plot main line graph
        for _, grp in data.groupby('id'):
            ax.plot(grp['time'], grp['outcome'], label=grp['type'].iloc[0], color=raw_color[0], linewidth=0.5)

        # Add points for last dot
        data_last_dot = data[data["last_dot"] == 1]
        ax.scatter(data_last_dot["time"], data_last_dot["outcome"], color=raw_color[2], s=10)

        # Customize legend
        ax.legend(title="", labels=set_labels, loc='lower center', ncol=labels_ncol)

        # Set labels and title
        ax.set_xlabel(xlab if xlab is not None else "Time")
        ax.set_ylabel(ylab if ylab is not None else "Outcome")
        ax.set_title(main if main is not None else "Raw Data", fontsize=cex_main, fontweight='bold')

        # Rotate x-axis labels
        plt.xticks(rotation=angle, ha='right' if x_h == 1 else 'center')

        # Set x-ticks labels if necessary
        if not isinstance(time_label[0], (int, float)):
            ax.set_xticks(show)
            ax.set_xticklabels(time_label)

        # Set limits for y-axis
        if ylim is not None:
            ax.set_ylim(ylim)

        # Display grid based on gridOff
        if gridOff == 0:
            ax.grid(True)

        # Show plot
        plt.tight_layout()
        plt.show()

        
        


    ##############################
    ## Separate Plot (by.group == True)
    ##############################

    # Setting main title if not provided
    main = main if main is not None else "Raw Data"

    # Setting legend labels
    set_labels = legend_labs if legend_labs is not None and len(legend_labs) == 2 else ["Control", "Treatment"]

    # Control group plotting
    if 1 in unit_type:
        co_pos = np.where(unit_type == 1)[0]
        Nco = len(co_pos)

        data1 = pd.DataFrame({
            "time": np.tile(time[show], Nco),
            "outcome": Y[show, co_pos].flatten(),
            "type": ["co"] * (Nco * len(show)),
            "id": np.repeat(range(1, Nco + 1), len(show))
        })

        limits1 = ["co", "tr"]
        colors1 = raw_color[:2]
        main1 = "Always Under Control"

    # Treatment group plotting
    if 2 in unit_type:
        tr_pos = np.where(unit_type == 2)[0]
        Ntr = len(tr_pos)

        data2 = pd.DataFrame({
            "time": np.tile(time[show], Ntr),
            "outcome": Y[show, tr_pos].flatten(),
            "type": ["tr"] * (Ntr * len(show)),
            "id": np.repeat(range(1, Ntr + 1), len(show))
        })

        limits2 = ["co", "tr"]
        colors2 = raw_color[:2]
        main2 = "Always Under Treatment"

    # Reversal group plotting
    if 3 in unit_type:
        rv_pos = np.where(unit_type == 3)[0]
        Nrv = len(rv_pos)

        if outcome_type == "continuous":
            D_plot = D_old.copy()
            D_plot[D_plot == 0] = np.nan
            D_plot[I == 0] = np.nan

            D_rv = D_plot[:, rv_pos]
            Y_rv = Y[:, rv_pos]

            Y_trt = Y_rv * D_rv
            Y_trt_show = Y_trt[show, :]
            time_trt_show = time[show]

            ut_time, ut_id = [], []
            for i in range(Nrv):
                if np.sum(np.isnan(Y_trt_show[:, i])) != len(show):
                    ut_id.extend([i] * (len(show) - np.sum(np.isnan(Y_trt_show[:, i]))))
                    ut_time.extend(time_trt_show[~np.isnan(Y_trt_show[:, i])])

            T1_0 = np.sum(T1 == 0)
            T1_1 = np.sum(T1 == 1)
            N_T1_1 = T1_1
            N_T1_0 = Nrv * len(show) + len(ut_id) - N_T1_1

            data3 = pd.DataFrame({
                "time": np.concatenate([np.tile(time[show], Nrv), ut_time]),
                "outcome": np.concatenate([Y[show, rv_pos].flatten(), Y_trt_show[~np.isnan(Y_trt_show)].flatten()]),
                "type": np.concatenate([["co"] * (Nrv * len(show)), ["tr"] * len(ut_id)]),
                "last_dot": np.concatenate([["0"] * N_T1_0, ["1"] * N_T1_1]),
                "id": np.concatenate([np.repeat(range(1, Nrv + 1), len(show)), ut_id])
            })

            # Compute last_dot for treated and control separately
            data3_tr = data3[data3["type"] == "tr"]
            data3_tr["idtimes"] = data3_tr.groupby("id").cumcount() + 1
            data3_tr["last_dot"] = np.where(data3_tr.groupby("id")["idtimes"].transform('max') == 1, 1, 0)

            data3_co = data3[data3["type"] == "co"]
            data3_co["idtimes"] = data3_co.groupby("id").cumcount() + 1
            data3 = pd.concat([data3_co, data3_tr])

        else:  # Categorical outcome
            data3 = pd.DataFrame({
                "time": np.tile(time[show], Nrv),
                "outcome": Y[show, rv_pos].flatten(),
                "type": obs_missing[show, :].flatten(),
                "id": np.repeat(range(1, Nrv + 1), len(show))
            })

            data3["type"] = pd.Categorical(data3["type"], categories=[-1, 1], ordered=True).rename_categories(["co", "tr"])

        limits3 = ["co", "tr"]
        colors3 = raw_color[:2]
        main3 = "Treatment Status Changed"

    # Subplot function for each type
    def subplot(data, limits, labels, colors, main, outcome_type, theme_bw):
        fig, ax = plt.subplots()

        # Theme
        if theme_bw:
            sns.set(style="whitegrid")
        else:
            sns.set(style="darkgrid")

        # Continuous outcome plotting
        if outcome_type == "continuous":
            sns.lineplot(x="time", y="outcome", hue="type", style="type", data=data, ax=ax, palette=colors)
            ax.scatter(data[data["last_dot"] == 1]["time"], data[data["last_dot"] == 1]["outcome"], color=raw_color[2], s=10)
        else:  # Categorical outcome plotting
            sns.scatterplot(x="time", y="outcome", hue="type", data=data, ax=ax, palette=colors)

        # Title and axis labels
        ax.set_title(main if main != "" else "Raw Data", fontsize=cex_main, fontweight="bold")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        # X-axis ticks rotation
        plt.xticks(rotation=angle)

        # Legend
        ax.legend(labels=labels, loc="upper right")

        # Return the plot
        return fig, ax

    # Plotting based on different unit types
    if by_group_side is False:
        if len(np.unique(unit_type)) == 1:
            if 1 in unit_type:
                fig, ax = subplot(data1, limits1, set_labels, colors1, main1, outcome_type, theme_bw)
            elif 2 in unit_type:
                fig, ax = subplot(data2, limits2, set_labels, colors2, main2, outcome_type, theme_bw)
            elif 3 in unit_type:
                fig, ax = subplot(data3, limits3, set_labels, colors3, main3, outcome_type, theme_bw)
        else:
            # Handling for multiple unit types (e.g., combination of control, treatment, and reversal)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            if 1 in unit_type:
                subplot(data1, limits1, set_labels, colors1, main1, outcome_type, theme_bw)
            if 2 in unit_type:
                subplot(data2, limits2, set_labels, colors2, main2, outcome_type, theme_bw)
            if 3 in unit_type:
                subplot(data3, limits3, set_labels, colors3, main3, outcome_type, theme_bw)
        plt.tight_layout()
        plt.show()

        
        
        
        

    ############# Treatment Status ###############
    elif type == "treat":

        if xlab is None:
            xlab = index[1]
        elif xlab == "":
            xlab = None

        if ylab is None:
            ylab = index[0]
            if collapse_history:
                ylab = "Number of Units"
        elif ylab == "":
            ylab = None

        if main is None:
            if collapse_history:
                main = "Unique Treatment Histories"
            else:
                if ignore_treat == 0:
                    main = "Treatment Status"
                else:
                    main = "Missing Values"
        elif main == "":
            main = None

        units = np.repeat(np.arange(1, N + 1)[::-1], TT)
        period = np.tile(np.arange(1, TT + 1), N)

        m = obs_missing[show, :].values
        all_vals = np.unique(m[~np.isnan(m)])

        col = []
        breaks = []
        label = []

        if not d_bi and ignore_treat == 0:  # >2 treatment levels

            tr_col = [
                "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494",
                "#FAFAD2", "#ADFF2F", "#87CEFA", "#1874CD", "#00008B"
            ]

            if treat_type == "discrete":
                for i in range(n_levels):
                    breaks.append(d_levels[i])
                    label.append(str(d_levels[i]))
                col = tr_col[:n_levels]

            else:
                interval = (max(d_levels) - min(d_levels)) / 4
                m = np.where((m >= min(d_levels)) & (m < min(d_levels) + interval), min(d_levels), m)
                m = np.where((m >= min(d_levels) + interval) & (m < min(d_levels) + 2 * interval), min(d_levels) + interval, m)
                m = np.where((m >= min(d_levels) + 2 * interval) & (m < min(d_levels) + 3 * interval), min(d_levels) + 2 * interval, m)
                m = np.where(m >= max(d_levels), max(d_levels), m)

                breaks = [
                    min(d_levels), min(d_levels) + interval, min(d_levels) + 2 * interval,
                    min(d_levels) + 3 * interval, max(d_levels)
                ]
                col = ["#c6dbef", "#4292c6", "#1f78b4", "#08519c", "#042b53"]
                label = [str(b) for b in breaks]
                treat_type = "discrete"

            if -200 in all_vals:
                col.append("#FFFFFF")
                breaks.append(-200)
                label.append("Missing")

        else:  # binary treatment indicator

            if 0 in all_vals:  # have pre and post: general DID type data

                if -1 in all_vals:
                    col.append("#B0C4DE")
                    breaks.append(-1)
                    label.append("Controls")

                col.append("#4671D5")
                breaks.append(0)
                label.append("Treated (Pre)")

                if 1 in all_vals:
                    col.append("#06266F")
                    breaks.append(1)
                    label.append("Treated (Post)")

            else:

                if -1 in all_vals:
                    col.append("#B0C4DE")
                    breaks.append(-1)
                    label.append("Under Control" if ignore_treat == 0 else "Observed")

                if 1 in all_vals:
                    col.append("#06266F")
                    breaks.append(1)
                    label.append("Under Treatment")

            if -200 in all_vals:
                col.append("#FFFFFF")
                breaks.append(-200)
                label.append("Missing")

            if len(id) > 1 and ignore_treat == 0 and d_bi:

                if by_timing:
                    co_seq = np.where(unit_type == 1)[0]
                    tr_seq = np.setdiff1d(np.arange(N), co_seq)
                    dataT0 = pd.DataFrame({"id": tr_seq, "T0": T0, "co_total": co_total})
                    dataT0 = dataT0.sort_values(by=["T0", "co_total", "id"])
                    tr_seq = dataT0["id"].values
                    missing_seq = np.concatenate([tr_seq, co_seq])

                    m = m[:, missing_seq]
                    id = id[missing_seq]

        if color is not None:
            if treat_type == "discrete":
                if len(col) == len(color):
                    print(f"Specified colors in the order of: {', '.join(label)}.")
                    col = color
                else:
                    raise ValueError(f"Length of 'color' should be equal to {len(col)}.")

        if legend_labs is not None:
            if treat_type == "discrete":
                if len(legend_labs) != len(label):
                    print("Incorrect number of labels in the legends. Using default.\n")
                else:
                    print(f"Specified labels in the order of: {', '.join(label)}.")
                    label = legend_labs

        res = m.flatten()
        data = pd.DataFrame({"units": units, "period": period, "res": res})

        if leave_gap == 0:
            data = data.dropna()

        data["res"] = data["res"].astype("category")

        if m.shape[1] >= 200:
            if axis_lab == "both":
                axis_lab = "time"
            elif axis_lab == "unit":
                axis_lab = "off"

        if background is not None:
            grid_color = border_color = background_color = legend_color = background
        else:
            grid_color = border_color = background_color = legend_color = "grey90"

        id = id[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the heatmap (equivalent to geom_tile)
        pivot_table = data.pivot("units", "period", "res")
        sns.heatmap(pivot_table, ax=ax, cmap=col, cbar_kws={'label': 'Treatment level'}, linewidths=0.1 if not grid_off else 0)

        # Customize labels and title
        ax.set_xlabel(xlab, fontsize=cex_lab, labelpad=8)
        ax.set_ylabel(ylab, fontsize=cex_lab, labelpad=8)
        ax.set_title(main, fontsize=cex_main, pad=8)

        # Customize axis and text appearance
        ax.tick_params(axis='both', which='major', labelsize=cex_axis)
        plt.xticks(rotation=angle, ha='right' if x_h == 1 else 'center', fontsize=cex_axis_x)
        plt.yticks(fontsize=cex_axis_y)

        # Remove grid lines if necessary
        if grid_off:
            ax.grid(False)

        # Customize the borders
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(0.5)

        # Customize background colors
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        # Legend settings
        if n_levels < 3:
            ax.get_legend().set_title(None)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(breaks)
        cbar.set_ticklabels(label)

        cbar.ax.set_facecolor(legend_color)

        # Handle axis labels based on user input
        if axis_lab == "both":
            ax.set_xticks(T_b)
            ax.set_xticklabels(time_label[T_b])
            ax.set_yticks(N_b)
            ax.set_yticklabels(id[N_b])
        elif axis_lab == "unit":
            ax.set_xticks(T_b)
            ax.set_xticklabels([])
            ax.set_yticks(N_b)
            ax.set_yticklabels(id[N_b])
        elif axis_lab == "time":
            ax.set_xticks(T_b)
            ax.set_xticklabels(time_label[T_b])
            ax.set_yticks(N_b)
            ax.set_yticklabels([])
        elif axis_lab == "off":
            ax.set_xticks(np.arange(1, len(show)+1))
            ax.set_xticklabels([])
            ax.set_yticks(np.arange(1, N+1))
            ax.set_yticklabels([])

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()
        
        
        
        

    # Determine styles for plotting
    if len(style) == 0:
        if treat_type == "discrete" and outcome_type == "continuous":
            ystyle = "line"
            Dstyle = "bar"
        elif treat_type == "discrete" and outcome_type == "discrete":
            ystyle = "bar"
            Dstyle = "bar"
        elif treat_type == "continuous" and outcome_type == "discrete":
            ystyle = "bar"
            Dstyle = "line"
        elif treat_type == "continuous" and outcome_type == "continuous":
            ystyle = "line"
            Dstyle = "line"
    else:
        if len(style) == 2:
            ystyle = style[0]
            Dstyle = style[1]
        elif len(style) == 1:
            ystyle = style[0]
            Dstyle = style[0]
        else:
            raise ValueError('Length of "style" should not be larger than 2.')

    # Axes labels
    if xlab is None:
        xlab = Dname
    elif xlab == "":
        xlab = None

    if ylab is None:
        ylab = Yname
    elif ylab == "":
        ylab = None

    # Prepare data
    plot_data = pd.DataFrame({
        "time": data["time"],
        "outcome": data["outcome"],
        "treatment": data["treatment"],
        "id": data["id"],
        "input_id": data["input.id"]
    })

    # Aggregate data by time (mean over time)
    data_means = plot_data.groupby("time").mean().reset_index()

    # Plot settings
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if theme_bw:
        plt.style.use('grayscale')
    
    ax1.set_xlabel(xlab if xlab else "")
    ax1.set_ylabel(ylab if ylab else "")

    # Plot outcome
    if ystyle == "line":
        ax1.plot(data_means["time"], data_means["outcome"], label=ylab, color="dodgerblue", linewidth=2)
    elif ystyle == "bar":
        ax1.bar(data_means["time"], data_means["outcome"], label=ylab, color="dodgerblue", alpha=0.7)
    
    # Define secondary axis (treatment)
    ax2 = ax1.twinx()
    ax2.set_ylabel(Dname if Dname else "")

    if Dstyle == "line":
        ax2.plot(data_means["time"], data_means["treatment"], label=Dname, color="lightsalmon", linewidth=2, linestyle="--")
    elif Dstyle == "bar":
        ax2.bar(data_means["time"], data_means["treatment"], label=Dname, color="lightsalmon", alpha=0.3)

    # Set y-limits if provided
    if ylim is not None:
        ax1.set_ylim(ylim[0])
        ax2.set_ylim(ylim[1])

    # Title and legend
    plt.title(f"{ylab} vs {xlab}", fontsize=16, fontweight='bold')
    fig.tight_layout()  # Adjust layout
    
    fig.legend(loc=legend_pos)
