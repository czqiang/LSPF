    begin


    f                          = addfile("./piomas/ease2_nh25km_grid.nc", "r")
    lat2d                      = f->lat
    lon2d                      = f->lon


    ; piomas
    f                          = addfile("./piomas/piomas.2023.nc", "r")
    sic_pio                    = f->piomas(:,:,:,0)
    sit_pio                    = f->piomas(:,:,:,1) * 10.

    sic_pio@lat2d              = lat2d
    sic_pio@lon2d              = lon2d

    sit_pio@lat2d              = lat2d
    sit_pio@lon2d              = lon2d

    sic_pio                    = mask(sic_pio, sic_pio .gt. 0.15, True)
    sit_pio                    = mask(sit_pio, sic_pio .gt. 0.15, True)

    ;;avg
    avg_sic_pio                = dim_avg_n_Wrap(sic_pio, 0)
    avg_sit_pio                = dim_avg_n_Wrap(sit_pio, 0)
    
    



    ; vae
    f                          = addfile("./piomas/vae.0-6.nc", "r")
    sic_vae                    = f->vae_ice(:,:,:,0)
    sit_vae                    = f->vae_ice(:,:,:,1) * 10. 

    sic_vae@lat2d              = lat2d
    sic_vae@lon2d              = lon2d

    sit_vae@lat2d              = lat2d
    sit_vae@lon2d              = lon2d

    sic_vae                    = mask(sic_vae, sic_vae .gt. 0.15, True)
    sit_vae                    = mask(sit_vae, sic_vae .gt. 0.15, True)

    ;;avg
    avg_sic_vae                = dim_avg_n_Wrap(sic_vae, 0)
    avg_sit_vae                = dim_avg_n_Wrap(sit_vae, 0)
    
    


    ; dif
    avg_sic_dif                = avg_sic_vae
    avg_sic_dif                = avg_sic_vae - avg_sic_pio 
    

    avg_sit_dif                = avg_sit_vae
    avg_sit_dif                = avg_sit_vae - avg_sit_pio 




    ;; plot
    wks                        = gsn_open_wks("pdf", "figure2")
    ; gsn_define_colormap(wks, "cmp_b2r")

    res                        = True
    res@gsnDraw                = False
    res@gsnFrame               = False
    res@gsnPolar               = "NH"
    res@mpMinLatF              = 55.
    res@cnFillOn               = True
    res@cnLinesOn              = False
    res@cnFillMode             = "RasterFill"


    plots                      = new(6, graphic)


    ;; sic

    res@cnFillPalette          = "wh-bl-gr-ye-re"

    res@gsnStringFontHeightF   = 0.022
    res@lbLabelFontHeightF     = 0.018

    res@cnLevelSelectionMode   = "ManualLevels"
    res@cnMinLevelValF         = 0.05
    res@cnMaxLevelValF         = 0.95
    res@cnLevelSpacingF        = 0.05

    res@gsnLeftString          = "(a)"
    res@gsnCenterString        = "PIOMAS Mean SIC 2023"
    res@gsnRightString         = "[0-1]"
    plots(0)                   = gsn_csm_contour_map_polar(wks, avg_sic_pio, res)


    res@gsnLeftString          = "(b)"
    res@gsnCenterString        = "VAE Generated Mean SIC"
    res@gsnRightString         = "[0-1]"
    plots(1)                   = gsn_csm_contour_map_polar(wks, avg_sic_vae, res)


    res@cnFillPalette          = "BlueWhiteOrangeRed"

    res@cnLevelSelectionMode   = "ManualLevels"
    res@cnMinLevelValF         = -0.14
    res@cnMaxLevelValF         =  0.14
    res@cnLevelSpacingF        =  0.02

    res@gsnLeftString          = "(c)"
    res@gsnCenterString        = "DIFF SIC: VAE - PIOMAS"
    res@gsnRightString         = "[0-1]"
    plots(2)                   = gsn_csm_contour_map_polar(wks, avg_sic_dif, res)



    ;; sit

    res@cnFillPalette          = "wh-bl-gr-ye-re"

    res@cnLevelSelectionMode   = "ManualLevels"
    res@cnMinLevelValF         = 0.2
    res@cnMaxLevelValF         = 3.2
    res@cnLevelSpacingF        = 0.2


    res@gsnLeftString          = "(d)"
    res@gsnCenterString        = "PIOMAS Mean SIT 2023"
    res@gsnRightString         = "[m]"
    plots(3)                   = gsn_csm_contour_map_polar(wks, avg_sit_pio, res)

    res@gsnLeftString          = "(e)"
    res@gsnCenterString        = "VAE Generated Mean SIT"
    res@gsnRightString         = "[m]"
    plots(4)                   = gsn_csm_contour_map_polar(wks, avg_sit_vae, res)

    res@cnFillPalette          = "BlueWhiteOrangeRed"

    res@cnLevelSelectionMode   = "ManualLevels"
    res@cnMinLevelValF         = -0.7
    res@cnMaxLevelValF         =  0.7
    res@cnLevelSpacingF        =  0.1


    res@gsnLeftString          = "(f)"
    res@gsnCenterString        = "DIFF SIT: VAE - PIOMAS"
    res@gsnRightString         = "[m]"
    plots(5)                   = gsn_csm_contour_map_polar(wks, avg_sit_dif, res)



    resp                       = True
    resp@gsnFrame              = False

    gsn_panel(wks, plots, (/2,3/), resp)

    frame(wks)

    end