begin

    file_path = "/home/xeon/2020/fix/cs2smos/dst/"


    day_start = 32
    day_end = 106

    cell_area = 25. * 25.

    days = day_end - day_start + 1

    ice = new((/days, 3/), "float")


    do n = day_start, day_end

        nnn = sprinti("%0.3i", n)

        file_name = file_path + "f" + nnn + ".nc"

        f = addfile(file_name, "r")
        sic = f->obs(0,:,:,0)
        sit = f->obs(0,:,:,1) * 10.


        counter = 0
        volume = 0.
        avgsit = 0.


        do i = 0, 431
        do j = 0, 431

            if (sic(i,j) .gt. 0.15) then
                counter = counter + 1
                volume = volume + cell_area * sit(i,j)
                avgsit = avgsit + sit(i,j)
            end if

        end do
        end do

        ice(n-day_start,0) = cell_area * counter / 1.e+6
        ice(n-day_start,1) = volume / 1.e+6        
        ice(n-day_start,2) = avgsit / counter

    end do

    write_matrix(ice, "3f10.4", False)
     

end