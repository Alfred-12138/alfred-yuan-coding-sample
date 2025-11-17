//------------------------------------------------------------
// Project : Low-Carbon City Pilot & Innovation
// Purpose : End-to-end workflow for an empirical paper:
//           data import → cleaning → DID analysis → robustness → plots
// Author  : Alfred Yuan
// Date    : 2025-03-16
//------------------------------------------------------------

clear
**Set the path to the current working dictionary
cd "`c(current_do_file)'"


*=========== Import data  ============

**Import original data
import excel "data/raw_data.xlsx", sheet("Sheet1") firstrow
**Save as dta file
save "data/raw_data.dta",replace



/*==================================================
                    Data Cleaning
==================================================*/

*=========== 1. Delete cities with many missing values  ============
local dropcities "莱芜市 三沙市 儋州市 毕节市 巢湖市 铜仁市 日喀则市 "  ///
	"昌都市 林芝市 山南市 海东市 吐鲁番市 哈密市 拉萨市"

foreach c in `dropcities' {
	display "`c'"
    drop if city == "`c'"
}



*=========== 2. Generate/Process variables  ============

**Generate staggered DID dummy variables
gen lccp_treat = (lccp_treatment_year != 0)
gen icp_treat = (lccp_treatment_year != 0)

gen lccp = (lccp_treat == 1 & year >= lccp_treatment_year + 1) // LCCP Policy
gen icp = (icp_treat == 1 & year >= icp_treatment_year + 1) //ICP Policy

**Process/Generate the control variables and dependent variables
gen sci_spend_share = science_spending / gdp /1000
gen loan_share = loan / gdp / 1000000
gen educ = college_students / population / 10000
gen road_per_capita = road_area / population
replace industry_structure = industry_structure / 100
replace green_rate = green_rate / 100

gen l_pergdp = ln(gdp_per_capita)
gen l_perroad = ln(road_per_capita)

replace embeddedness = embeddedness / 1000

**Remove cities with too many dependent variables with values equal to zero
bysort city_no: egen zero_embeddedness = total(embeddedness==0)
bysort city_no: egen zero_complexity = total(complexity==0)
gen both_zero = (zero_embeddedness > 8 & zero_complexity > 8)
drop if both_zero == 1

** Store the control variable lists as a global variable
global control educ l_pergdp urbanization sci_spend_share industry_structure  ///
	green_rate energy_structure loan_share l_perroad

*=========== 3. Winsor all continous variables  ============

foreach var of varlist $control embeddedness complexity {
    if strpos(" $control ", " `var' ") {
        winsor2 `var', cut(1 99) replace
    }
    else {
        winsor2 `var', cut(0 99) replace // Dependent variables only winsor tail
    }
}

*=========== 4. Standarize dependent variables ============

foreach var of varlist embeddedness complexity {
	egen z_`var' = std(`var')
	drop `var'
	rename z_`var' `var'
}

** Save the cleaned dataset
save "data/cleaned_data.dta", replace


/*==================================================
               Pre-estimate Analysis
==================================================*/
capture mkdir "results"
capture mkdir "results/pre-estimate_analysis"
*=========== 1. Descriptive Statistics ===========
eststo clear
estpost summarize embeddedness complexity $control
esttab using "results/pre-estimate_analysis/descriptive_results.rtf", ///
    cells("mean(fmt(3)) sd(fmt(3)) min(fmt(1)) max(fmt(1)) count(fmt(0))") ///
    label title("Descriptive Statistics") replace

*=========== 2. Correlation Analysis ===========
logout, save("results/pre-estimate_analysis/correlation_results.rtf") ///
	word replace: pwcorr embeddedness complexity $control, sig star(0.05)
	

*=========== 3. The trend of changes in the dependent variable ===========
foreach var of varlist embeddedness complexity {
	preserve
		collapse (mean) `var'_mean = `var', by(year lccp_treat)
		twoway ///
			(line `var'_mean year if lccp_treat==0, ///
			sort lpattern(solid)  lwidth(medthick)) ///
			(line `var'_mean year if lccp_treat==1, ///
			sort lpattern(dash)   lwidth(medthick)), ///
			legend(order(1 "Control" 2 "Treatment") pos(6) ring(0)) ///
			xtitle("Year") ytitle("Mean of `var'") ///
			title("`var' over time by group") ///
			scheme(s1mono) ///
			name(trend_by_group, replace)
		
		graph export "results/pre-estimate_analysis/`var'_trend.png", ///
		replace width(1600) height(1000) 
		graph close trend_by_group
	restore
}

/*==================================================
                    Main Analysis
==================================================*/
capture mkdir "results/main_results"

*=========== 1. Baseline Model Regression ===========
foreach var of varlist embeddedness complexity {
	local mode_emb = cond(`var'==embeddedness, "replace", "append")
	
	**Baseline model without control
	reghdfe `var' lccp , absorb(city_no year) cluster(city_no)
	**Export the regression results
	outreg2 using "results/main_results/baseline_regression.doc", ///
		word `mode_emb' se bdec(3) sdec(2) ctitle("`var'") ///
		addtext(Controls, NO, City FE, YES, Year FE, YES)
	
	**Baseline model with control
	reghdfe `var' lccp $control , absorb(city_no year) cluster(city_no) 
	
	outreg2 using "results/main_results/baseline_regression.doc", ///
		word append se bdec(3) sdec(2) ctitle("`var'") ///
		addtext(Controls, YES, City FE, YES, Year FE, YES)
	

*=========== 2. Crowding out/Complementary Effect ===========

	reghdfe `var' lccp##icp $control , absorb(city_no year) cluster(city_no)
	outreg2 using "results/main_results/complementary_effect.doc", ///
		word `mode_emb' se bdec(3) sdec(2) ctitle("`var'") ///
		keep(lccp##icp) addtext(Controls, YES, Fix Effects, YES)
}

*=========== 3.	Moderating effect ===========

**3.1 Moderating effect (Digitalization)

**Process relevent variables
gen per_internet = internet / population * 100
gen per_tele = telephone / population * 100
gen per_tele_service = tele_service / population
gen info_employee_share = info_employee / population

global digital_cov per_internet per_tele per_tele_service info_employee_share

**Principle component anaylysis
xtset city_no year
factortest $digital_cov
pca $digital_cov
estat kmo
predict pc1 pc2, score
gen digit = (pc1 * 0.6804 + pc2 * 0.1487) / 0.8292 //Generate digitalization variable
winsor2 digit, cuts(0 95) replace

egen digit_std = std(digit) //Standarize the variable
drop digit
rename digit_std digit

**Pctiles setup
local pcts 10 25 50 75 90
_pctile digit, p(`pcts')

local i = 1
foreach p of local pcts {
	local digit`p' = r(r`i')
	local dx`p' : display %9.3f `digit`p''
	local ++i
}

**Moderating effect analysis
foreach var of varlist embeddedness complexity {
	local mode_emb = cond(`var'==embeddedness, "replace", "append")
	
	**Moderating effect estimation
	reghdfe `var' i.lccp##c.digit $control , absorb(city_no year) cluster(city_no)
	
	outreg2 using "results/main_results/Moderating_digit.doc", ///
		word `mode_emb' se bdec(3) sdec(2) ctitle("`var' (Digitaization)") ///
		keep(i.lccp##c.digit) addtext(Controls, YES, City FE, YES, Year FE, YES)
	
	testparm 1.lccp##c.digit
	
	**Marginal effect map at different quantiles
	margins, dydx(1.lccp) at(digit=(`digit10' `digit25' `digit50' ///
	`digit75' `digit90'))

	marginsplot, xdimension(digit) recast(connected) plotopts(msymbol(O) ///
	msize(med) lwidth(med)) ciopts(recast(rline) lpattern(dash) lwidth(thin) ) ///
	   xlabel(`dx10' "P10" `dx25' "P25" `dx50' "P50" `dx75' "P75" ///
	   `dx90' "P90", angle(0)) ///
		xline(`digit10' `digit25' `digit50' `digit75' `digit90', lpattern(dot)) ///
		yline(0, lpattern(solid) lwidth(med)) ///
		ytitle("Marginal Effects of LCCP on `var'") ///
		xtitle("Digitalization (centered) Quantiles") ///
		title("Moderating Effect by Digitalization") ///
		scheme(s1mono) ///
		name(digitalization_`var', replace)
		
	graph export "results/main_results/moderating_digitalization_`var'.png", replace
	graph close digitalization_`var'
}
	


**3.2 Moderating effect (Environment regulation)

gen l_so2 = -ln(so2_emission)
gen l_dust = -ln(dust)
gen green_per_capita = green_area / population


global env_cov l_so2 l_dust industrial_water_rate garbage_rate

**PCA
xtset city_no year
factortest $env_cov
pca $env_cov
estat kmo
predict env_pc1 env_pc2, score
gen environment = (env_pc1 * 0.5109 + env_pc2 * 0.3245) / 0.8353
winsor2 environment, cuts(0 95) replace

** Standarize the moderating variable
egen environment_std = std(environment)
drop environment
rename environment_std environment


**Quantiles setup
local pcts 10 25 50 75 90
_pctile environment, p(`pcts')

local i = 1
foreach p of local pcts {
	local environment`p' = r(r`i')
	local env`p' : display %9.3f `environment`p''
	local ++i
}

**Moderating effect analysis
foreach var of varlist embeddedness complexity {
	local mode_emb = cond(`var'==embeddedness, "replace", "append")
	
	**Moderating effect estimation
	reghdfe `var' i.lccp##c.environment $control , absorb(city_no year) ///
		cluster(city_no)
	
	outreg2 using "results/main_results/Moderating_environment.doc", ///
		word `mode_emb' se bdec(3) sdec(2) ctitle("`var' (Environment)") ///
		keep(i.lccp##c.environment) addtext(Controls, YES, City FE, YES, Year FE, YES)
	
	testparm 1.lccp##c.environment
	
	**Marginal effect map at different quantiles
	margins, dydx(1.lccp) at(environment=(`environment10' `environment25' ///
		`environment50' `environment75' `environment90'))

	marginsplot, name(environment_`var', replace) ///
	xdimension(environment) recast(connected) plotopts(msymbol(O) ///
	msize(med) lwidth(med)) ciopts(recast(rline) lpattern(dash) lwidth(thin) ) ///
	   xlabel(`env10' "P10" `env25' "P25" `env50' "P50" `env75' "P75" ///
	   `env90' "P90", angle(0)) ///
		xline(`environment10' `environment25' `environment50' ///
		`environment75' `environment90', lpattern(dot)) ///
		yline(0, lpattern(solid) lwidth(med)) ///
		ytitle("Marginal Effects of LCCP on `var'") ///
		xtitle("Environment (centered) Quantiles") ///
		title("Moderating Effect by Environment") ///
		scheme(s1mono)
	graph export "results/main_results/moderating_environment_`var'.png", replace
	graph close environment_`var'
}


*=========== 4.	Heterogeneity analysis ===========	

replace region = 1 if region == 4
replace city_size = 1 if city_size < 4
replace city_size = 0 if city_size == 4

**4.1-4.5 Region, City size, Key Environmental Protection Cities, 
**        Resource-based Cities, Direct vs. Indirect Pilot Cities

foreach category of varlist region city_size env_city resource_city lccp_prov {
	
	if "`category'" == "region" {
        local glist 1 2 3
    }
    else {
        local glist 1 0
    }
	
	foreach var of varlist embeddedness complexity {
		foreach g of local glist {
			local mode_emb = cond(`g'==1 & `var'==embeddedness, "replace", "append")
			
			if "`category'" == "lccp_prov" {
                local cond_expr "lccp_treat == 0 | `category' == `g'"
            }
            else {
                local cond_expr "`category' == `g'"
            }
			
			
			reghdfe `var' lccp $control if `cond_expr', ///
				absorb(city_no year) cluster(city_no)

			outreg2 using "results/main_results/hetero_`category'.doc", ///
				`mode_emb' se bdec(3) sdec(2) keep(lccp) ///
				ctitle("`var' (`category' `g')") ///
				addtext(Controls, YES, City FE, YES, Year FE, YES)
		}
	}
}

**Delate all txt files
local txtfiles: dir "results/main_results" files "*.txt"
foreach txt in `txtfiles' {
    erase `"results/main_results/`txt'"'
}

/*==================================================
                 Robustness checks
==================================================*/
capture mkdir "results/robustness_checks"

*=========== 1.	Testing for parallel trend (Event Study) ===========	

** Generate dummy variables relative to policy time
gen lccp_policy = year - (lccp_treatment_year + 1)  

forvalues i=7(-1)1{
  gen lccp_pre`i'=(lccp_policy==-`i' & lccp_treat == 1)
}

gen lccp_current = (lccp_policy==0 & lccp_treat == 1)

forvalues i=1/11{
  gen lccp_post`i'=(lccp_policy==`i' & lccp_treat == 1)
}

replace lccp_pre1 = 0 //set 1 year before policy as baseline


**Event study
foreach var of varlist embeddedness complexity {
	
	reghdfe `var' lccp_pre* lccp_current lccp_post* $control , ///
	absorb(city_no year) cluster(city_no)
	
	local coeflabs
	forvalues k = 7(-1)1 {
		local coeflabs `coeflabs' lccp_pre`k' = "`= -`k''"
	}
	local coeflabs `coeflabs' lccp_current = "0"
	forvalues k = 1/11 {
		local coeflabs `coeflabs' lccp_post`k' = "`k'"
	}
	**Coefficient graph
	coefplot, baselevels omitted ///
		keep(lccp_pre* lccp_current lccp_post*) ///
		vertical ///
		coeflabels(`coeflabs') ///
	   yline(0)                             ///
	   xline(7,lpattern(dash)) ///
	   ytitle(`"Coefficient"',size(large)) ///
	   ylabel(,angle(horizontal) labsize(medium)) ///
	   xtitle(`"Years Relative to LCCP Implementation"',size(large)) ///
	   xlabel(,labsize(medium)) ///
	   ciopts(lpattern(dash) recast(rcap) msize(medium))                 ///
	   scheme(s1mono)                   ///
	   levels(95)  ///
	   graphregion(color(white)) ///
	   plotregion(lcolor(white) style(none)) ///
	   recast(connected) msymbol(O) msize(medium) lwidth(medium) ///
	   name(event_study_twfe_`var', replace)

	graph export "results/robustness_checks/event_study_twfe_`var'.png", replace
	graph close event_study_twfe_`var'
}

save "data/cleaned_data.dta",replace

*=========== 2.	Placebo test ===========	

capture mkdir "results/robustness_checks/placebo_test"
foreach var of varlist embeddedness complexity {
	* Choose pseudo-treated cities and rerun the regression
	clear
	set matsize 5000
	mat b = J(500,1,0)
	mat se = J(500,1,0)
	mat p = J(500,1,0)
	forvalues i=1/500 {
		*Set pseudo-treated cities numbers
		local Ntreat 117

		*Generate startyear
		use "data/cleaned_data.dta", clear
		keep city_no
		bysort city_no: keep if _n==1
		set seed `=100000 + `i''
		
		local p2010 = 57/117
		local p2012 = 27/117
		local p2017 = 33/117

		set seed `=100500 + `i''

		*Each pseudo-treated city is assigned to a real treatment year
		gen u = runiform()
		gen startyear = .
		replace startyear = 2010 if u < `p2010'
		replace startyear = 2012 if u >= `p2010' & u < `= `p2010' + `p2012''
		replace startyear = 2017 if u >= `= `p2010' + `p2012''
		drop u
		
		tempfile matchyear_city
		save `matchyear_city', replace

		*Select the pseudo-treated cities list
		use "data/cleaned_data.dta", clear
		keep city_no
		bysort city_no: keep if _n==1
		set seed `=200267 + `i''
		quietly count
		local poolN = r(N)
		local take  = min(`Ntreat', `poolN')
		sample `take', count
		gen groupnew = 1
		tempfile treat_cities
		save `treat_cities', replace

		* 3) Combine the whole sample
		use "data/cleaned_data.dta", clear
		merge m:1 city_no using `treat_cities'
		replace groupnew = 0 if missing(groupnew)
		drop _merge
		merge m:1 city_no using `matchyear_city'
		drop _merge

		xtset city_no year
		gen time = (year > startyear)
		gen did  = time * groupnew

		reghdfe `var' did $control, absorb(city_no year) cluster(city_no)
		mat b[`i',1]  = _b[did]
		mat se[`i',1] = _se[did]
		mat p[`i',1]  = 2*ttail(e(df_r), abs(_b[did]/_se[did]))
	}

	svmat b, names(coef)
	svmat se, names(se)
	svmat p, names(pvalue)

	drop if pvalue1 == .
	label var pvalue1 pvalue
	label var coef1 estimated_coefficients
	save "results/robustness_checks/placebo_test/placebo.dta", replace

	twoway ///
		(scatter pvalue1 coef1, yaxis(1) ///
			ytitle("{stSans:P value}", axis(1)) ///
			yline(0.05, lp(shortdash)) msymbol(smcircle_hollow) mcolor(gs8)) ///
		(kdensity coef1, yaxis(2) ///
			ytitle("{stSans:Density}", axis(2)) lcolor(black) lwidth(medthick)), ///
			xtitle("{stSans:Estimated Coefficient}") ///
		xlabel(-0.2(0.1)0.4, format(%03.2f) angle(0)) ///
		xline(0.345, lp(shortdash)) ///
		title("{stSans:Placebo Test}") ///
		legend(off) ///
		scheme(s1mono) ///
		name(placebo_test_`var', replace)

	graph export "results/robustness_checks/placebo_test/placebo_`var'.png", replace
	graph close placebo_test_`var'

}



*=========== 3.	Propensity Score Matching ===========	
capture mkdir "results/robustness_checks/psm"

* PSM based on each implement year (3 waves)
local lccp_policy_years 2010 2012 2017
foreach pyear of local lccp_policy_years {

	use "data/cleaned_data.dta", clear
	
	*Keep only the implement year data for psm
	keep if year == `pyear'
	gen current_treat = (lccp_treatment_year == `pyear')
	drop if lccp_treatment_year != 0 & lccp_treatment_year != `pyear'

	*PSM
	psmatch2 current_treat $control , logit com neighbor(2) caliper(0.05) ties
	pstest $control

	keep if !missing(_weight)
	tempfile base
	save `base', replace

	* 1) Organize controls
	preserve
		keep if current_treat==1
		keep _id _n1 _n2 _nn
		
		rename _n1 n1
		rename _n2 n2
		reshape long n, i(_id) j(match)        
		keep if match <= _nn                
		drop match
		rename n member_id
		gen group_id = _id
		keep group_id member_id
		tempfile controls_long_`pyear'
		save controls_long_`pyear', replace
	restore

	* 2) Add treated to group with controls
	preserve
		keep if current_treat==1
		gen group_id  = _id
		gen member_id = _id
		keep group_id member_id
		tempfile treated_rows
		save treated_rows_`pyear', replace
	restore
	use treated_rows_`pyear', clear
	append using controls_long_`pyear'

	rename member_id _id
	merge m:1 _id using `base', nogenerate
	rename _id member_id

	keep city_no group_id member_id _treated current_treat _weight

	order group_id member_id _treated current_treat
	sort group_id _treated member_id

	save match_`pyear'_unique, replace
	use "data/cleaned_data.dta", replace
	merge m:n city_no using match_`pyear'_unique.dta, nogenerate

	keep if !missing(group_id)
	gen match_wave = `pyear'
	save "results/robustness_checks/psm/lccp_match_`pyear'", replace
}

use "results/robustness_checks/psm/lccp_match_2010", clear
append using "results/robustness_checks/psm/lccp_match_2012"
append using "results/robustness_checks/psm/lccp_match_2017"

**Generate unique group id
sort match_wave group_id
egen new_group_id = group(match_wave group_id), label
drop group_id
rename new_group_id group_id

save "results/robustness_checks/psm/matched_all", replace

foreach var of varlist embeddedness complexity {
	local mode_emb = cond(`var'==embeddedness, "replace", "append")
	
	reghdfe `var' lccp $control [aw=_weight], absorb(group_id year) ///
		cluster(city_no)
	outreg2 using "results/robustness_checks/psm/psm_results.doc", ///
		word `mode_emb' se bdec(3) sdec(2) keep(lccp) ctitle("`var'") ///
		addtext(Controls, YES, City FE, YES, Year FE, YES)

}

**Delate intermediate data files
local dtafiles: dir . files "*.dta"
foreach dta in `dtafiles' {
    erase `"`dta'"'
}

*=========== 4.	Other robustness checks ===========
use "data/cleaned_data.dta",clear

**4.1  Exclude the Four Municipalities.
preserve
	drop if city_no == 1101 | city_no == 3101 | city_no == 1201 | city_no ==5001
	
	foreach var of varlist embeddedness complexity {
		local mode_emb = cond(`var'==embeddedness, "replace", "append")

		reghdfe `var' lccp , absorb(city_no year) cluster(city_no)
		outreg2 using "results/robustness_checks/exclude_4cities.doc", ///
			word `mode_emb' se bdec(3) sdec(2) keep(lccp) ctitle("`var'") ///
			addtext(Controls, YES, City FE, YES, Year FE, YES)
	}
restore

**4.2 Add confusion event (Carbon Emission Pilot Policy)

gen etp = ((year > etp_treatment_year) & (etp_treatment_year != 0))
foreach var of varlist embeddedness complexity {
	local mode_emb = cond(`var'==embeddedness, "replace", "append")
	
	reghdfe `var' lccp etp $control, absorb(city_no year) cluster(city_no)
	outreg2 using "results/robustness_checks/add_ept.doc", ///
		word `mode_emb' se bdec(3) sdec(2) keep(lccp) ctitle("`var'") ///
		addtext(Controls, YES, City FE, YES, Year FE, YES)
}

**4.3 Alternative dependent variable
replace embeddedness_total = embeddedness_total /1000
foreach var of varlist embeddedness_total complexity_total {
    winsor2 `var', cut(0 99) replace
	egen z_`var' = std(`var')
	drop `var'
	rename z_`var' `var'
	
}

foreach var of varlist embeddedness_total complexity_total {
	local mode_emb = cond(`var'==embeddedness_total, "replace", "append")
	
	reghdfe `var' lccp etp $control, absorb(city_no year) cluster(city_no)
	outreg2 using "results/robustness_checks/alternative_dependent_variable.doc", ///
		word `mode_emb' se bdec(3) sdec(2) keep(lccp) ctitle("`var'") ///
		addtext(Controls, YES, City FE, YES, Year FE, YES)
}
	
**Delate all txt files
local txtfiles: dir "results/robustness_checks" files "*.txt"
foreach txt in `txtfiles' {
    erase `"results/robustness_checks/`txt'"'
}
	
**4.3 Multi-period Policy Heterogeneity (did imputation)

replace lccp_treatment_year =. if lccp_treatment_year == 0
replace icp_treatment_year =. if icp_treatment_year == 0
gen id = _n


foreach var of varlist embeddedness complexity {
	
	did_imputation `var' id year lccp_treatment_year, controls($control) ///
		horizon(0/11) autosample fe(city_no year) cluster(city_no) ///
		pretrend(7) tol(1e-1)
	
	local coeflabs
	forvalues k = 7(-1)1 {
		local coeflabs `coeflabs' pre`k' = "`= -`k''"
	}
	forvalues k = 0/11 {
		local coeflabs `coeflabs' tau`k' = "`k'"
	}
	**Coefficient graph
	coefplot, baselevels omitted ///
		keep(pre* tau*) ///
		order(pre7 pre6 pre5 pre4 pre3 pre2 pre1 tau0 tau1 tau2 tau3 ///
		tau4 tau5 tau6 tau7 tau8 tau9 tau10 tau11) ///
		vertical ///
		coeflabels(`coeflabs') ///
	   yline(0)        ///
	   xline(7,lpattern(dash)) ///
	   ytitle(`"Coefficient"',size(large)) ///
	   ylabel(,angle(horizontal) labsize(medium)) ///
	   xtitle(`"Years Relative to LCCP Implementation"',size(large)) ///
	   xlabel(,labsize(medium)) ///
	   ciopts(lpattern(dash) recast(rcap) msize(medium))                 ///
	   scheme(s1mono)                   ///
	   levels(95)  ///
	   graphregion(color(white)) ///
	   plotregion(lcolor(white) style(none)) ///
	   recast(connected) msymbol(O) msize(medium) lwidth(medium) ///
	   title("Borusyak et al. (2021) imputation estimator", size(large)) ///
	   name(event_study_imputation, replace)

	graph export "results/robustness_checks/event_study_imputation_`var'.png", replace
	graph close event_study_imputation
}
