begin

;; grid
f             = addfile("./piomas/ease2_nh25km_grid.nc", "r")
lat2d         = f->lat
lon2d         = f->lon


;; sis2
f             = addfile("./sis2/sis2.nc", "r")
sic           = f->sis2(0,:,:,0)
sit           = f->sis2(0,:,:,1) * 10.

sic@lat2d     = lat2d
sic@lon2d     = lon2d

sit@lat2d     = lat2d
sit@lon2d     = lon2d

sic           = mask(sic, sic .gt. 0.15, True)
sit           = mask(sit, sic .gt. 0.15, True)


;; vae
f             = addfile("./sis2/samples.nc", "r")
sic_ens       = f->particles(:,:,:,0)
sit_ens       = f->particles(:,:,:,1) * 10.


delete(sic_ens@_FillValue)
sic_ens@lat2d = lat2d
sic_ens@lon2d = lon2d


delete(sit_ens@_FillValue)
sit_ens@lat2d = lat2d
sit_ens@lon2d = lon2d


sic_ens       = mask(sic_ens, sic_ens .gt. 0.15, True)
sit_ens       = mask(sit_ens, sic_ens .gt. 0.15, True)


avg_sic       = dim_avg_n_Wrap(sic_ens, 0)
avg_sit       = dim_avg_n_Wrap(sit_ens, 0)


std_sic = dim_stddev_n_Wrap(sic_ens, 0) 
std_sit = dim_stddev_n_Wrap(sit_ens, 0)




;; plot
wks = gsn_open_wks("x11", "x")

res                        = True
res@gsnDraw                = False
res@gsnFrame               = False
res@gsnPolar               = "NH"
res@mpMinLatF              = 50.
res@cnFillOn               = True
res@cnLinesOn              = False
res@cnFillMode             = "RasterFill"


plots                      = new(6, graphic)


;; sic
res@cnFillPalette          = "cmp_b2r"

res@gsnStringFontHeightF   = 0.022
res@lbLabelFontHeightF     = 0.018

res@cnLevelSelectionMode   = "ManualLevels"
res@cnMinLevelValF         = 0.05
res@cnMaxLevelValF         = 0.95
res@cnLevelSpacingF        = 0.05

res@gsnLeftString          = "(a)"
res@gsnCenterString        = "SIS2 Model SIC"
res@gsnRightString         = "[0-1]"
plots(0)                   = gsn_csm_contour_map_polar(wks, sic, res)

res@gsnLeftString          = "(b)"
res@gsnCenterString        = "VAE Generated Mean SIC"
res@gsnRightString         = "[0-1]"
plots(1)                   = gsn_csm_contour_map_polar(wks, avg_sic, res)


res@cnFillPalette          = "WhiteBlueGreenYellowRed"

res@cnLevelSelectionMode   = "ManualLevels"
res@cnMinLevelValF         = 0.01
res@cnMaxLevelValF         = 0.08
res@cnLevelSpacingF        = 0.01

res@gsnLeftString          = "(c)"
res@gsnCenterString        = "VAE Generated SIC (Std)"
res@gsnRightString         = "[0-1]"
plots(2)                   = gsn_csm_contour_map_polar(wks, std_sic, res)



res@cnFillPalette          = "cmp_b2r"

res@cnLevelSelectionMode   = "ManualLevels"
res@cnMinLevelValF         = 0.2
res@cnMaxLevelValF         = 4.0
res@cnLevelSpacingF        = 0.2


res@gsnLeftString          = "(d)"
res@gsnCenterString        = "SIS2 Model SIT"
res@gsnRightString         = "[m]"
plots(3)                   = gsn_csm_contour_map_polar(wks, sit, res)

res@gsnLeftString          = "(e)"
res@gsnCenterString        = "VAE Generated Mean SIT"
res@gsnRightString         = "[m]"
plots(4)                   = gsn_csm_contour_map_polar(wks, avg_sit, res)


res@cnFillPalette          = "WhiteBlueGreenYellowRed"

res@cnLevelSelectionMode   = "ManualLevels"
res@cnMinLevelValF         = 0.02
res@cnMaxLevelValF         = 0.18   
res@cnLevelSpacingF        = 0.02

res@gsnLeftString          = "(f)"
res@gsnCenterString        = "VAE Generated SIT (Std)"
res@gsnRightString         = "[m]"
plots(5)                   = gsn_csm_contour_map_polar(wks, std_sit, res)

resp                       = True
resp@gsnFrame              = False

gsn_panel(wks, plots, (/2,3/), resp)

frame(wks)


end 