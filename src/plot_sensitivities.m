clc, clear, close all

% Importing data
fdm_data = importdata('fdm.dat');
cvm_data = importdata('cvm.dat');
sacvm_data = importdata('sacvm.dat');

% Plotting
figure;
hold on
ph1=plot(fdm_data(:,1), fdm_data(:,2), '-k','LineWidth',1.5);
plot(fdm_data(:,1), fdm_data(:,3), '-k','LineWidth',1.5)
plot(fdm_data(:,1), fdm_data(:,4), '-k','LineWidth',1.5)
ph2=plot(cvm_data(:,1), cvm_data(:,2), '--r','LineWidth',1.5);
plot(cvm_data(:,1), cvm_data(:,3), '--r','LineWidth',1.5)
plot(cvm_data(:,1), cvm_data(:,4), '--r','LineWidth',1.5)
ph3=plot(sacvm_data(:,1), sacvm_data(:,2), '-.b','LineWidth',1.5);
plot(sacvm_data(:,1), sacvm_data(:,3), '-.b','LineWidth',1.5)
plot(sacvm_data(:,1), sacvm_data(:,4), '-.b','LineWidth',1.5)

% Adding legend and improving some aesthetics
set(gca,'box','on')
legend([ph1 ph2 ph3],'FDM','CVM','SACVM')
xlabel('$x$','Interpreter','LaTeX')
ylabel('Sensitivities')

