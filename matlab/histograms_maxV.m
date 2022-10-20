parks = {'Vilanova', 'SantSadurni', 'SantBoi', 'QuatreCamins',...
                      'Cerdanyola','Granollers','Martorell','Mollet',...
                      'SantQuirze','PratDelLlobregat'};
str_parks = {'Vilanova', 'Sant Sadurni', 'Sant Boi', 'Quatre Camins',...
                      'Cerdanyola','Granollers','Martorell','Mollet',...
                      'Sant Quirze','Prat del Llobregat'};

idxOK = [1:6 8 10];
parksOK = {parks{idxOK}};
str_parksOK = {str_parks{idxOK}};

str = '../data/maxOccupancy.csv';
data = readtable(str);
parkLims = data.Var2(idxOK);
    
h = figure(1); clf;
set(h,'PaperType','A4');
set(h,'PaperOrientation','portrait');
set(h,'PaperUnits','centimeters');
set(h,'PaperPosition',[.6 0 20 20]); 
set(h,'PaperSize',[20 20]);
FontSize = 18;

for p = 1:numel(parksOK)
    
    subplot(4,2,p)

    binsize = 10;
    ymax = 50;
    max_reached = 0;

    if p==1
        %plot(parkLims(p)*ones(10,1), linspace(0,ymax,10),'--','linewidth',2,'color','r');
        grid on;
        set(gca,'ylim',[0,ymax]);
        hold on;
        xt = get(gca,'xtick');
    else 
        %plot(parkLims(p)*ones(10,1), linspace(0,ymax,10),'--','linewidth',2,'color','r');
        grid on;
        set(gca,'ylim',[0,ymax]);
        hold on;
        set(gca,'xtick',xt);
    end
    set(gca,'xlim',[0 max(parkLims)+10]);
    %set(gca,'yscale','log')

    % weekday
%    str = ['../data/' parksOK{p} '_WD_maxV.csv'];
    str = ['../data/' parksOK{p} '_WD_maxV2.csv'];
    Ns = csvread(str,1);
    data = Ns(:,2);
    [a1,b1] = hist(data,0:binsize:max(parkLims));
    max_reached = sum(data==parkLims(p));
    
    % friday
    hold on;
%    str = ['../data/' parksOK{p} '_FR_maxV.csv'];
    str = ['../data/' parksOK{p} '_FR_maxV2.csv'];
    Ns = csvread(str,1);
    data = Ns(:,2);
    max_reached = max_reached+sum(data==parkLims(p));
    [a2,b2] = hist(data,0:binsize:max(parkLims));

    % weekend
    hold on;
%    str = ['../data/' parksOK{p} '_WE_maxV.csv'];
    str = ['../data/' parksOK{p} '_WE_maxV2.csv'];
    Ns = csvread(str,1);
    data = Ns(:,2);
    max_reached = max_reached+sum(data==parkLims(p));
    [a3,b3] = hist(data,0:binsize:max(parkLims));
    
    title(str_parksOK{p});
    set(gca,'ylim',[0 max([a1,a2,a3])]);
    fprintf('%s:\tmax reached %d days\n',parksOK{p},max_reached);

    bh= bar(b3,[a1;a2;a3]);
    bh(1).EdgeAlpha=0;
    bh(2).EdgeAlpha=0;
    bh(3).EdgeAlpha=0;
    bh(1).FaceColor=[1,0.65,0];
    bh(2).FaceColor='b';
    bh(3).FaceColor=[0,0.5,0];
    bh(1).BarWidth = 5;
    bh(2).BarWidth = 5;
    bh(3).BarWidth = 5;
    plot(parkLims(p)*ones(10,1)+6, linspace(0,ymax,10),'-','linewidth',2,'color','r');

%     f0=fill([b3 b3(end:-1:1)],[zeros(1,numel(b3)) a1(end:-1:1)],[1,0.65,0]);
%     f0.EdgeAlpha=0;
%     f0.FaceAlpha=.5;
%     f1=fill([b3 b3(end:-1:1)],[zeros(1,numel(b3)) a2(end:-1:1)],'b');
%     f1.EdgeAlpha=0;
%     f1.FaceAlpha=.5;
%     f2=fill([b3 b3(end:-1:1)],[zeros(1,numel(b3)) a3(end:-1:1)],[0,0.5,0]);
%     f2.EdgeAlpha=0;
%     f2.FaceAlpha=.5;
    if p==1
        xt = get(gca,'xtick');
    else 
        set(gca,'xtick',xt);
    end
    if p==1
        hl = legend('WD','FR','WE','Capacity');
        set(hl,'location','northeast');
    end
end

saveas(h,'fig_Ns.pdf');