% savegif template

figure('units','normalized','outerposition',[0 0 1 1])
filename='name.gif';

for rayn=1:512
    
    clf
    % PLOT
    drawnow
    
    % Capture the plot as an image 
      frame = getframe(gca); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if rayn == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','DelayTime',0.1,'WriteMode','append'); 
      end 
end