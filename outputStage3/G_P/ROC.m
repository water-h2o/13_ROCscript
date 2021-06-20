filesHere = dir('*.xlsx');
in_addresses = {filesHere.name};

for i = 1:2:length(in_addresses)
    
    leg = {};

    figure(i);
    clf;
    hold on
    grid on
    plot([0.001:0.001:1],[0.001:0.001:1], 'r', 'LineWidth', 1.2)
    %plot([0.5 1],[0.5 0], 'r', 'LineWidth', 2, 'HandleVisibility', 'off')
    hold off

    leg{length(leg)+1} = 'tp = fp';
    
    
    in_address_FP = in_addresses{i};
    in_address_TP = in_addresses{i+1};
    
    disp(in_address_FP)
    disp(in_address_TP)
    
    fname_curr_FP = in_address_FP;
    fname_curr_TP = in_address_TP;
    
    curr_M_in_FP = readmatrix(fname_curr_FP);
    curr_M_in_FP_sz = size(curr_M_in_FP);
    
    curr_M_in_TP = readmatrix(fname_curr_TP);
    curr_M_in_TP_sz = size(curr_M_in_TP);
    
    FP = flipud(curr_M_in_FP(2:(end-2),:));
    TP = flipud(curr_M_in_TP(2:(end-2),:));
    
    d_s = 1.628./sqrt(curr_M_in_FP((end-1),:));
    e_s = 1.628./sqrt(curr_M_in_FP((end),:));
    
    D_s = repmat(d_s,[(curr_M_in_FP_sz(1)-3) 1]);
    E_s = repmat(d_s,[(curr_M_in_FP_sz(1)-3) 1]);
    
    conf_top_FP = FP - E_s;
    conf_top_TP = TP + D_s;
    
    conf_bot_FP = FP + E_s;
    conf_bot_TP = TP - D_s;
    
    FP_interp = 0:0.001:1;
    FP_interp_interval = FP_interp(2) - FP_interp(1);
    
    conf_top_TP_interp = zeros(length(FP_interp), length(d_s));
    conf_bot_TP_interp = zeros(length(FP_interp), length(d_s));
    
    disp('generated all confidence bands')
    
    for t = 1:length(d_s)
        
        disp(['t = ' num2str(t)])
        
        curr_top_FP = conf_top_FP(:,t);
        curr_top_TP = conf_top_TP(:,t);    
        fst_pos = find(curr_top_FP > 0 ,1);
        curr_top_FP = curr_top_FP(fst_pos:end);
        curr_top_TP = curr_top_TP(fst_pos:end);
        disp(['length(curr_top_FP) = ' num2str(length(curr_top_FP))])
        disp(['length(curr_top_TP) = ' num2str(length(curr_top_TP))])
        curr_top_FP = [0; curr_top_FP; 1];
        curr_top_TP = [curr_top_TP(1); curr_top_TP; curr_top_TP(end)];
        
        curr_bot_FP = conf_bot_FP(:,t);
        curr_bot_TP = conf_bot_TP(:,t);
        fst_one = find(curr_bot_FP > 1,1);
        curr_bot_FP = curr_bot_FP(1:(fst_one - 1));
        curr_bot_TP = curr_bot_TP(1:(fst_one - 1));
        curr_bot_FP = [0; curr_bot_FP; 1];
        curr_bot_TP = [curr_bot_TP(1); curr_bot_TP; curr_bot_TP(end)];
        
        figure(i);
        hold on
        
        plot(FP(:,t),TP(:,t),'Color',[0.7 0.7 0.7], ...
             'HandleVisibility','off')
        plot(curr_top_FP, curr_top_TP, '--', 'Color',[1 0.7 1],...
             'HandleVisibility','off')
        plot(curr_bot_FP, curr_bot_TP, '--', 'Color',[0.7 0.7 1],...
             'HandleVisibility','off')
        
        hold off
        
        d_X_top            = -999;
        d_X_bot            = -999;
        idx_curr_top_L_old = -999; 
        idx_curr_bot_L_old = -999;
        
        for j = 1:(length(FP_interp)-1)
        
            %disp(['j = ' num2str(j)])
            
            idx_curr_top_L = find(curr_top_FP > FP_interp(j) ,1)-1;
            idx_curr_top_R = idx_curr_top_L + 1;
            
            D_X_top_in = curr_top_FP(idx_curr_top_R) ...
                       - curr_top_FP(idx_curr_top_L);
            D_Y_top_in = curr_top_TP(idx_curr_top_R) ...
                       - curr_top_TP(idx_curr_top_L);
            
            DY_DX_top_in = D_Y_top_in / D_X_top_in;
                   
            d_X_top = d_X_top + FP_interp_interval;
            d_X_top = d_X_top ...
                    - (idx_curr_top_L ~= idx_curr_top_L_old)*d_X_top;
            
            curr_top_TP_in_L = curr_top_TP(idx_curr_top_L);
            conf_top_TP_interp(j,t) = curr_top_TP_in_L ...
                                    + (DY_DX_top_in * d_X_top);
                                
            idx_curr_top_L_old = idx_curr_top_L;
            
            idx_curr_bot_L = find(curr_bot_FP > FP_interp(j) ,1)-1;
            idx_curr_bot_R = idx_curr_bot_L + 1;
            
            D_X_bot_in = curr_bot_FP(idx_curr_bot_R) ...
                       - curr_bot_FP(idx_curr_bot_L);
            D_Y_bot_in = curr_bot_TP(idx_curr_bot_R) ...
                       - curr_bot_TP(idx_curr_bot_L);
            
            DY_DX_bot_in = D_Y_bot_in / D_X_bot_in;
                   
            d_X_bot = d_X_bot + FP_interp_interval;
            d_X_bot = d_X_bot ...
                    - (idx_curr_bot_L ~= idx_curr_bot_L_old)*d_X_bot;
            
            curr_bot_TP_in_L = curr_bot_TP(idx_curr_bot_L);
            conf_bot_TP_interp(j,t) = curr_bot_TP_in_L ...
                                    + (DY_DX_bot_in * d_X_bot);
            
            idx_curr_bot_L_old = idx_curr_bot_L;
        end
    end
    
    FP_mean = mean(FP,2);
    TP_mean = mean(TP,2);
    
    conf_top_TP_Lenvel = min(conf_top_TP_interp,[],2); 
    conf_bot_TP_Uenvel = max(conf_bot_TP_interp,[],2);
    
    AUC_top = trapz(FP_interp(1:(end-1)), conf_top_TP_Lenvel(1:(end-1)));
    AUC_avg = trapz(FP_mean, TP_mean);
    AUC_bot = trapz(FP_interp(1:(end-1)), conf_bot_TP_Uenvel(1:(end-1)));
    AUC_up  = AUC_top - AUC_avg;
    AUC_dn  = AUC_bot - AUC_avg;
    
    AUC_up_str = num2str(round(AUC_up,2));
    AUC_dn_str = num2str(round(abs(AUC_dn),2));
    AUC_avg_str = num2str(round(AUC_avg,2));
    
    figure(i)
    
    hold on
    
    plot(FP_interp(1:(end-1)), conf_top_TP_Lenvel(1:(end-1)), ...
        'm','LineWidth',1.5)
    leg{length(leg)+1} = 'upper 90% CL bound';
    plot(FP_interp(1:(end-1)), conf_bot_TP_Uenvel(1:(end-1)), ...
        'b','LineWidth',1.5)
    leg{length(leg)+1} = 'lower 90% CL bound';
    
    fill(cat(2,FP_interp(1:(end-1)),fliplr(FP_interp(1:(end-1)))),...
         cat(1,conf_top_TP_Lenvel(1:(end-1)),flipud(conf_bot_TP_Uenvel(1:(end-1)))),...
         'g', ...
         'LineStyle', 'none', 'FaceAlpha', 0.2)
    leg{length(leg)+1} = '90% CL band';
     
    plot(FP_mean,TP_mean, 'Color',[0.3 0.3 0.3],'LineWidth',1.5)
    leg{length(leg)+1} = 'average ROC curve'; 
    
    legend(leg,'Location','southeast')
    
    fill([0.595 0.96 0.96 0.595],[0.31 0.31 0.395 0.395], 'w', ...
         'HandleVisibility','off')
    
    str = ['$$\mathsf{AUC} = \mathsf{' AUC_avg_str...
           '}_{-\mathsf{' AUC_dn_str...
           '}}^{+\mathsf{' AUC_up_str...
           '}}\,,\,\mathsf{90}\%\,\mathsf{CL} $$'];
    text(0.61,0.35,str,'Interpreter','latex','FontSize',7)
    
    xlim([0 1])
    ylim([0 1])
    xlabel('fp')
    ylabel('tp')
    
    hold off
end

figure(1)

hold on
title('90% ROC confidence band for ISO scenario performance')
hold off

set(gcf, 'PaperUnits', 'inches');
x_width = 4;
y_width = 3;
set(gca,'fontsize', 7);
set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
%set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf,  ['./imgs_ROC/ISO_abs_perform.png']);
disp(['saved ISO figure'])

figure(3)

hold on
title('90% ROC confidence band for b2b scenario performance')
hold off

set(gcf, 'PaperUnits', 'inches');
x_width = 4;
y_width = 3;
set(gca,'fontsize', 7);
set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
%set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf,  ['./imgs_ROC/b2b_abs_perform.png']);
disp(['saved b2b figure'])