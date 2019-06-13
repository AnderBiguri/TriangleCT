function isin=incube(point,vmax,vmin)


isin=all([point(1)>=vmin(1) point(1)<=vmax(1) ...
          point(2)>=vmin(2) point(2)<=vmax(2) ...
          point(3)>=vmin(3) point(3)<=vmax(3)]);

end