filesHere = dir('*.xlsx');
in_addresses = {filesHere.name};

for i = 1:2:length(in_addresses)
    
    figure(4*i + 1);
    clf;

%     figure(4*i + 2);
%     clf;
    
    
    leg_CDF = {};
    leg_CDF{length(leg_CDF)+1} = 'sample dCDF';
    leg_CDF{length(leg_CDF)+1} = 'smoothed avg. dCDF';
    
    leg_PDF = {};
    leg_PDF{length(leg_PDF)+1} = 'sample 0νββ dPDF';
    leg_PDF{length(leg_PDF)+1} = 'smoothed avg. 0νββ dPDF'
    leg_PDF{length(leg_PDF)+1} = 'sample 1e dPDF';
    leg_PDF{length(leg_PDF)+1} = 'smoothed avg. 1e dPDF'
    
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
    
    FP = curr_M_in_FP(2:(end-2),:);
    TP = curr_M_in_TP(2:(end-2),:);
    
    dims = size(FP);
    
    CDF_1e = 1 - FP;
    CDF_0v = 1 - TP;
    
    PDF_1e = zeros(dims);
    PDF_0v = zeros(dims);
    
    CDF_1e_avg = zeros(1,length(dims(1)));
    CDF_0v_avg = zeros(1,length(dims(1)));
    
    PDF_1e_avg = zeros(1,length(dims(1)));
    PDF_0v_avg = zeros(1,length(dims(1)));
    
    PDF_1e(1,:) = CDF_1e(1,:);
    PDF_0v(1,:) = CDF_0v(1,:);
    
    CDF_1e_avg(1) = mean(CDF_1e(1,:));
    CDF_0v_avg(1) = mean(CDF_0v(1,:));
    
    PDF_1e_avg(1) = mean(PDF_1e(1,:));
    PDF_0v_avg(1) = mean(PDF_0v(1,:));
    
    for k = 2:dims(1)
       
        PDF_1e(k,:) = CDF_1e(k,:) - CDF_1e((k-1),:);
        PDF_0v(k,:) = CDF_0v(k,:) - CDF_0v((k-1),:);
        
        CDF_1e_avg(k) = mean(CDF_1e(k,:));
        CDF_0v_avg(k) = mean(CDF_0v(k,:));
    
        PDF_1e_avg(k) = mean(PDF_1e(k,:));
        PDF_0v_avg(k) = mean(PDF_0v(k,:));
    end
    
    n_1e = repmat(curr_M_in_FP(end,:),[dims(1) 1]);
    n_0v = repmat(curr_M_in_FP(end-1,:),[dims(1) 1]);
    
    PDF_1e = PDF_1e.*n_1e;
    PDF_0v = PDF_0v.*n_0v;
    
    PDF_1e_avg = PDF_1e_avg*mean(curr_M_in_FP(end,:));
    PDF_0v_avg = PDF_0v_avg*mean(curr_M_in_FP(end-1,:));
    
    thresh = 0:0.01:1;
    
    for n = 1:dims(2)
    
        figure(4*i + 1);
        %clf;
        hold on
        plot(thresh, PDF_0v(:,n),'Color',[0 0.5 1 0.15], ...
             'HandleVisibility', 'off')
        title(['PDF\_0vbb ' in_address_FP(1:3)])

%         xlabel('τ')
%         ylabel('PDF_0_ν_β_β(τ)')
%         hold off

%         figure(4*i + 1);
%         %clf;
%         hold on

        plot(thresh, PDF_1e(:,n),'Color',[1 0 0 0.15], ...
             'HandleVisibility', 'off')
        title(['PDF\_1e ' in_address_FP(1:3)])
        xlabel('L_0_ν_β_β')
        ylabel('pdf(L_0_ν_β_β)')
        hold off

%         figure(4*i + 2);
%         %clf;
%         hold on
% 
%         plot(thresh, CDF_0v(:,n),'Color',[0.7 1 0.75], ...
%              'HandleVisibility', 'off')
%         title(['CDF\_0vbb ' in_address_FP(1:3)])
%         xlabel('τ')
%         ylabel('CDF_0_ν_β_β(τ)')
%         hold off
% 
%         figure(4*i + 2);
%         %clf;
%         hold on
% 
%         plot(thresh, CDF_1e(:,n),'Color',[1 0.7 0.7], ...
%              'HandleVisibility', 'off')
%         title(['CDF\_1e ' in_address_FP(1:3)])
%         xlabel('τ')
%         ylabel('CDF_1_e(τ)')
%         hold off

    end
    
    figure(4*i + 1);
    %clf;
    hold on
    plot([1.1 1.2], [0 0], 'Color',[0 0.5 1 0.15])
    plot(thresh, smooth(PDF_0v_avg,20),'Color',[0 0.5 1], ...
        'LineWidth', 2.5)
%    title(['PDF_0_ν_β_β , ' in_address_FP(1:3)])
% 
%     xlim([0 1])
%     ylim([0 1.2*max(smooth(PDF_0v_avg,20))])
%     legend(leg_PDF,'Location','northwest')
%     hold off
% 
%     figure(4*i + 1);
%     %clf;
%     hold on

    plot([1.1 1.2], [0 0], 'Color',[1 0 0 0.15])
    plot(thresh, smooth(PDF_1e_avg,20),'Color',[1 0 0], ...
        'LineWidth', 2.5)
    title(['pdf(L_0_ν_β_β)  ,  ' in_address_FP(1:3)])
    xlim([0 1])
    ylim([0 1.2*max([max(smooth(PDF_1e_avg,20)) ...
                     max(smooth(PDF_0v_avg,20))])])
    legend(leg_PDF,'Location','north')
    hold off
% 
%     figure(4*i + 2);
%     %clf;
%     hold on
% 
%     plot([1.1 1.2], [0 0], 'Color',[0.7 1 0.7])
%     plot(thresh, smooth(CDF_0v_avg,20),'Color',[0 1 0.75], ...
%         'LineWidth', 2.5)
%     title(['CDF_0_ν_β_β , ' in_address_FP(1:3)])
%     xlim([0 1])
%     legend(leg_CDF,'Location','northwest')
%     hold off
% 
%     figure(4*i + 2);
%     %clf;
%     hold on
% 
%     plot([1.1 1.2], [0 0], 'Color',[1 0.7 0.7])
%     plot(thresh, smooth(CDF_1e_avg,20),'Color',[1 0 0], ...
%         'LineWidth', 2.5)
%     title(['CDF_1_e , ' in_address_FP(1:3)])
%     xlim([0 1])
%     legend(leg_CDF,'Location','northwest')
%     hold off

end