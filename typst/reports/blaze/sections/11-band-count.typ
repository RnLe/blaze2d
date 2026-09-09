#import "../figures.typ": plot

= Scaling with Band Count <sec_band_count>

The band-count sweep requests 4 to 20 bands. The recorded 20-band TM solve times are 1.9 s for Blaze and 3.9 s for MPB. Workspace size depends on band count even where process-memory measurements appear flat.

#figure(
  plot("bands-bar"),
  caption: [
    Solve time against the number of requested bands for TM (left) and TE
    (right).
  ],
) <fig_bands_bar>
