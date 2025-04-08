
%fusulage(gövde)
fuse_h = 1;
fuse_l1 = 2;
fuse_l2 = 1;
fuse_l3 = 6;
fuse_w = 1;

%Tail(kuyruk)
tail_h= 0;
tail_l = 2;
tail_w = 5;
vertical_tail_h = 2;

%wing(kanat)
wing_l = 2;
wing_w = 10;
wing_h = 0;

%CG(center of gravity)
cg_x = fuse_l1+(wing_l/3);
cg_y = -1;
cg_z = 0;

%alanlar
S = 0.55; %m^2
b = 2.8956; %m
c = 0.18994; %m

%mass
mass = 13.5; %kg

%gravity
g = 9.81;

%hava yoğunluğu
rho = 1.2682;

%inertia matrix
J = [
    0.8244,0,-0.1204;
    0,1.135,0;
    -0.1204,0,1.759
];


%tahrik sistemi
S_prob = 0.2027; % m^2

K_motor = 80;
K_T_p = 0;
K_omega = 0;
e = 0.9;

%Longitudal aerodinamik katsatılar

M = 50;
alpha_0 = 0.4712;


CD = 0.03;
CL = 0.28;
Cm = -0.02338;

CD_a = 0.30;
CL_a = 3.45;
Cm_a = -0.38;

CD_q = 0;
CL_q = 0;
Cm_q = -3.6;

CD_delta_e = 0;
CL_delta_e =  -0.36;
Cm_delta_e = -0.5;

C_prob = 1.0;

CD_p = 0.0437;

%Lateral aerodinamik katsatılar

CY = 0;
Cl = 0;
Cn = 0;

CY_beta = -0.98;
Cl_beta = -0.12;
Cn_beta =  0.25;

CY_p = 0;
Cl_p = -0.26;
Cn_p = 0.022;

CY_r = 0;
Cl_r = 0.14;
Cn_r = -0.35;

CY_delta_a = 0;
Cl_delta_a = 0.08;
Cn_delta_a = 0.06;

CY_delta_r = -0.17;
Cl_delta_r = 0.105;
Cn_delta_r = -0.032;

%Atmospheric katsayılar

%dryden gust model parametreleri(WE HAVE TURBULANCE)
% mediumaltitude, light , turbulence(600 m alt)

L_u = 533; %m
L_v = 533; %m
L_w = 533; %m

sigma_u = 1.5; %m/s
sigma_v = 1.5; %m/s
sigma_w = 1.5; %m/s

%simulasyon ayarları
dt      = 0.01;                  % zaman adımı                    [s]
simTime = 30;                   % toplam simulasyon süresi       [s]
t = 0;
numSteps = simTime / dt;

% uçağın şeklini oluştur
points = CreateAircraftBody(fuse_h, fuse_l1, fuse_l2, fuse_l3, fuse_w, ...
                            tail_h, tail_l, tail_w, vertical_tail_h, ...
                            wing_l, wing_w, wing_h, cg_x, cg_y, cg_z);


points = reflect_points_y(points);  %noktaları ters yaptığımdan aynalayacağız :)


figure(1), clf
handle = [];
handle = VisualizeAircraft(points, 0, handle); % handle başlangıçta boş
xlabel('East');
ylabel('North');
zlabel('-Down');
view(32,47)  % set the vieew angle for figure
%axis([-100,100,-100,100,-50,50]);
grid on;
axis equal;
hold on;
view(3);
hold off;

% veriler için grafik olşuturma

figure(2);
tiledlayout(2, 3, 'TileSpacing', 'compact');

%1. Grafik: kuvvetler
nexttile;

fx_line = animatedline('Color', 'r');
fy_line = animatedline('Color', 'g');
fz_line = animatedline('Color', 'b');
title('long forces');
xlabel('Time (s)');
ylabel('Newton');
legend('Fx', 'Fy','Fz');

% 2. Grafik: momentler
nexttile;

l_line = animatedline('Color', 'r');
m_line = animatedline('Color', 'g');
n_line = animatedline('Color', 'b');
title('moments');
xlabel('Time (s)');
ylabel('Newton Metre');
legend('l', 'm','n');

%1. Grafik: hız vektörleri
nexttile;

u_line = animatedline('Color', 'r');
w_line = animatedline('Color', 'g');
v_line = animatedline('Color', 'b');
title('Velocity');
xlabel('Time (s)');
ylabel('m/s');
legend('u', 'w','x');

% 2. Grafik: açısal hızlar
nexttile;

p_line = animatedline('Color', 'r');
q_line = animatedline('Color', 'g');
r_line = animatedline('Color', 'b');
title('açısal hız');
xlabel('Time (s)');
ylabel('rad/s');
legend('p', 'q','r');

%1. Grafik: euler açıları
nexttile;

phi_line = animatedline('Color', 'r');
theta_line = animatedline('Color', 'g');
psi_line = animatedline('Color', 'b');

title('euler angles');
xlabel('Time (s)');
ylabel('deg');
legend('u', 'w','x');

% 2. Grafik: hava hızı
nexttile;

airspeed_line = animatedline('Color', 'r');

title('airspeed');
xlabel('Time (s)');
ylabel('m/s');
legend('p', 'q','r');

figure(3);

poz_line = animatedline('Color', 'g');
title('pozisyon NE');
xlabel('NORTH');
ylabel('EAST');
legend('N', 'E');

% Yeni bir UI figürü oluştur
fig = uifigure('Name', 'Command', 'Position', [100, 100, 300, 230]);

% Figür boyutlarını al
figWidth = fig.Position(3);
figHeight = fig.Position(4);

% Kaydırıcı boyutları
sliderWidth = 150;  % Genişlik
sliderHeight = 3;   % Yükseklik
sliderSpacing = 50; % Aralık

% Kaydırıcıların x pozisyonu (ortalanmış)
startX = (figWidth - sliderWidth) / 2;

% İlk kaydırıcının y başlangıç pozisyonunu hesapla
startY = (figHeight - (sliderSpacing * 3 + sliderHeight * 4)) / 2;

% aileren
sld1 = uislider(fig, ...
    'Position', [startX, startY, sliderWidth, sliderHeight], ...
    'Limits', [-30, 30], ...
    'Value', 0);

% elevtor
sld2 = uislider(fig, ...
    'Position', [startX, startY + sliderSpacing + sliderHeight, sliderWidth, sliderHeight], ...
    'Limits', [-30, 30], ...
    'Value', 0);

% throttle
sld3 = uislider(fig, ...
    'Position', [startX, startY + 2 * (sliderSpacing + sliderHeight), sliderWidth, sliderHeight], ...
    'Limits', [0, 1], ...
    'Value', 0);

% ruder
sld4 = uislider(fig, ...
    'Position', [startX, startY + 3 * (sliderSpacing + sliderHeight), sliderWidth, sliderHeight], ...
    'Limits', [-30, 30], ...
    'Value', 0);


%grafik ve log için
array_u = [];
array_w = [];
array_v = [];

array_pN = [];
array_PE = [];
array_PD = [];

array_phi = [];
array_theta = [];
array_psi = [];

array_p = []; 
array_q = []; 
array_r = []; 

array_fx = [];
array_fy = [];
array_fz = [];

array_l = [];
array_m = [];
array_n = [];

%atmospher

%sabit rüzgar hızı
w_n_s = 0.1;
w_e_s = 0.01;
w_d_s = 0.3;
Vws = [w_n_s;w_e_s;w_d_s];


%steady wind(WE HAVE TURBULANCE)
u_w_g = 0;
v_w_g = 0;
w_w_g = 0;
Vwg = [u_w_g;v_w_g;w_w_g];


u_w = 0;
v_w = 0;
w_w = 0;
Vw = [u_w;v_w;w_w];

%hız vektörü
u = 0; %m/s
v = 0; %m/s
w = 0; %m/s

%pozison
pN = 0;
pE = 0;
pD = 0;

%euler açıları
phi = deg2rad(0);
theta = deg2rad(0);
psi = deg2rad(0);

%açı rate
p = 0; %rad/s
q = 0; %rad/s
r = 0; %rad/s

%kuvvetler
fx = 0;
fy = 0;
fz = 0;

%açısal momentler
l = 0;
m = 0;
n = 0;

%aoa side slip
alpha = 0;
beta = 0;
Va = 0;

%kontrol yüzeyleri açısı
delta_e = 0;
delta_a = 0;
delta_r = 0;

%throttle yüzdesi
delta_throttle = 0;

for i = 1:numSteps-1

    delta_a = deg2rad(sld1.Value);
    delta_e = deg2rad(sld2.Value);
    delta_throttle = sld3.Value;
    delta_r = deg2rad(sld4.Value);

    %Rigid-body dinamiği

    %inertiadan body frame dönüşüm matrisi
    R_b_v = calc_R_b_v(phi,theta,psi);
    
    %atmosfer
    %Hs = calc_Hs(L_u,L_v,L_w,sigma_u,sigma_v,sigma_w,Va,s);%gürültü
    %Hs = [1;1;1]; %şimdilik 0 yaptık
    V_b_w = calc_V_b_w(R_b_v,Vws);%rüzgar

    Va_NED = calc_airspeed_NED(V_b_w,u,v,w);%hava hızı NED

    Va = calc_airspeed(Va_NED); %düz hava hızı

    alpha = calc_alpha(Va_NED); %hücüm açısı;

    beta = calc_beta(Va_NED,Va);

    %longitudal kuvvet hesapla
    aerodynamic_conff = calc_aerodynamic_forces(CD,CL,CD_a,CL_a,CD_q,CL_q,CD_delta_e,CL_delta_e,alpha);
    
    CX_alpha = aerodynamic_conff(1);
    CX_qalpha = aerodynamic_conff(2);
    CX_deltaE_alpha = aerodynamic_conff(3);
    CZ_alpha = aerodynamic_conff(4);
    CZ_qalpha = aerodynamic_conff(5);
    CZ_deltaE_alpha = aerodynamic_conff(6);

    longitudal_aero_forces = calc_longitudal_aero_forces(rho,Va,S,c,p,q,r,b,delta_e,delta_a,delta_r,...
                                        CX_alpha, CX_qalpha, CX_deltaE_alpha,...
                                        CZ_alpha,CZ_qalpha,CZ_deltaE_alpha,...
                                        CY,CY_beta,CY_r,CY_delta_a,CY_delta_r,CY_p...
                                        );

    gavitional_force = calc_gravitional_force(mass,g,phi,theta,psi);

    longitudal_thrust_forces = calc_thrust_forces(rho,Va,S_prob,C_prob,K_motor,delta_throttle);

    longitudal_forces = calc_longitudal_forces(gavitional_force,longitudal_aero_forces,longitudal_thrust_forces);
    
    fx = longitudal_forces(1);
    fy = longitudal_forces(2);
    fz = longitudal_forces(3);

    %hızları hesapla
    velocity_dot = calc_velocity_dot(p,q,r,u,w,v,mass,fx,fy,fz); %x,y,z de ki hız değişimi
    u = u + velocity_dot(1)*dt;
    w = w + velocity_dot(2)*dt;
    v = v + velocity_dot(3)*dt;


    %pozisyonu desapla
    p_dot = calc_ned_dot(u,w,v,phi,theta,psi,R_b_v); 
    pN = pN + p_dot(1)*dt;
    pE = pE + p_dot(2)*dt;
    pD = pD + p_dot(3)*dt;
    
    %momentleri hesapla 
    moment_aero_forces = calc_moment_aero_forces(rho,Va,S,c,b,p,q,beta,...
    Cl,Cl_beta,Cl_p,Cl_r,Cl_delta_a,delta_a,Cl_delta_r,delta_r,...
    alpha,Cm,Cm_a,Cm_q,Cm_delta_e,delta_e,...
    Cn,Cn_beta,Cn_p,Cn_r,Cn_delta_a,Cn_delta_r,r);

    moment_thrust_forces = calc_moment_thrust_forces(rho,Va,S,K_motor,delta_throttle,K_omega);

    moment_torque = calc_moment_torque(moment_aero_forces,moment_thrust_forces);
    
    l= moment_torque(1);
    m= moment_torque(2);
    n= moment_torque(3);
    
    %açısal hızları hesapla
    omega_dot = calc_omega_dot(J, p,q,r, l,m,n);
    p = p + omega_dot(1)*dt;
    q = q + omega_dot(2)*dt;
    r = r + omega_dot(3)*dt;

    %euler açılarını hesapla
    euler_dot = calc_euler_dot(p,q,r,phi,theta,psi);
    phi = phi + euler_dot(1)*dt;
    theta = theta + euler_dot(2)*dt;
    psi = psi + euler_dot(3)*dt;
    
    disp(rad2deg(phi));
    disp(rad2deg(theta));

    %görselleştirme
    points = translate_and_rotate_points(points,euler_dot(1),euler_dot(2),euler_dot(3),p_dot(1),p_dot(2),p_dot(3));%roll,pitch,yaw,x,y,z
    handle = VisualizeAircraft(points, i, handle); %animasyon
    
    addpoints(fx_line,t,fx);
    addpoints(fy_line,t,fy);
    addpoints(fz_line,t,fz);

    addpoints(u_line,t,u);
    addpoints(w_line,t,w);
    addpoints(v_line,t,v);

    addpoints(p_line,t,p);
    addpoints(q_line,t,q);
    addpoints(r_line,t,r);

    addpoints(phi_line,t,rad2deg(phi));
    addpoints(theta_line,t,rad2deg(theta));
    addpoints(psi_line,t,rad2deg(psi));

    addpoints(l_line,t,l);
    addpoints(m_line,t,m);
    addpoints(n_line,t,n);
    
    addpoints(airspeed_line,t,Va);

    addpoints(poz_line,pE,pN);
        
    array_u(i) = u;
    array_w(i) = w;
    array_v(i) = v;
    
    array_pN(i) = pN;
    array_pE(i) = pE;
    array_pD(i) = pD;
    
    array_phi(i) = phi;
    array_theta(i) = theta;
    array_psi(i) = psi;
    
    array_p(i) = p; 
    array_q(i) = q; 
    array_r(i) = r; 
    
    array_fx(i) = fx;
    array_fy(i) = fy;
    array_fz(i) = fz;
    
    array_l(i) = l;
    array_m(i) = m;
    array_n(i) = n;
    
    t = t + dt;
    pause(0.01);
end


%Small unmand aircraft kitabında ki ilk chapter uygulaması

clear; close all; clc % kayıtlı bütün verileri sıfırla

%rotasyon matrislerinin olduğu fonksiyon
function XYZ=rotate(XYZ,phi,theta,psi)%(ROLL,PTICH,YAW)
  % Rotasyon ve öteleme
  R_roll = [...
          1, 0, 0;...
          0, cos(phi), -sin(phi);...
          0, sin(phi), cos(phi)];
  R_pitch = [...
          cos(theta), 0, sin(theta);...
          0, 1, 0;...
          -sin(theta), 0, cos(theta)];
  R_yaw = [...
          cos(psi), -sin(psi), 0;...
          sin(psi), cos(psi), 0;...
          0, 0, 1];
  R = R_roll*R_pitch*R_yaw;
  % rotate vertices
  XYZ = R*XYZ;
end

%tüm noktaları rotasyon ve öteleme matrisleri ile çarpan fonksiyon
function translate_and_rotate_points_array = translate_and_rotate_points(points, phi, theta, psi, x, y, z)
    % Rotasyon matrisi için fonksiyonu çağır

    rotated_points_array = [];  % Çıktı dizisini başlat
    translation_vector = [x, y, z];  % Öteleme vektörü

    % Her bir nokta üzerinde rotasyon ve öteleme uygula
    for i = 1:size(points, 1)
        % Her nokta bir satırda [x, y, z] formatında
        point = points(i, :);
           
        % Noktayı 3x1 vektöre dönüştür
        XYZ = point(:);
        
        % Öteleme uygula
        translated_XYZ = XYZ + translation_vector(:);  % translation_vector 3x1 vektör
        
        % Rotasyonu uygula
        rotated_XYZ = rotate(translated_XYZ, phi, theta, psi);
        
        % Dönen noktayı diziye ekle
        rotated_points_array = [rotated_points_array; rotated_XYZ'];
    end

    % Çıktıyı atama: fonksiyonun çıktısı olarak rotated_points_array
    translate_and_rotate_points_array = rotated_points_array;
end

function reflect_points_array = reflect_points_y(points)
    % Rotasyon matrisi için fonksiyonu çağır

    reflect_points_array = [];  % Çıktı dizisini başlat

    % Y eksenine yansıma matrisi
    reflection_matrix_y = [-1, 0, 0; 0, 1, 0; 0, 0, 1];

    % Her bir nokta üzerinde rotasyon, öteleme ve yansıma uygula
    for i = 1:size(points, 1)
        % Her nokta bir satırda [x, y, z] formatında
        point = points(i, :);
        
        % Noktayı 3x1 vektöre dönüştür
        XYZ = point(:);
        
        % Y eksenine yansıma uygula
        reflected_XYZ = reflection_matrix_y * XYZ;  % Y eksenine yansıma işlemi
               
        % Dönen noktayı diziye ekle
        reflect_points_array = [reflect_points_array; reflected_XYZ'];
    end
end

%veriler girdilere göre geometrik şekili oluşturur
function outPoints = CreateAircraftBody(fuse_h,fuse_l1,fuse_l2,fuse_l3,fuse_w,tail_h,tail_l,tail_w,vertical_tail_h,wing_l,wing_w,wing_h,cg_x,cg_y,cg_z)

points_array = [];

%burun olşutur
p1 = [cg_x,cg_y,0];
p2 = [cg_x+(fuse_l1-fuse_l2),cg_y+(fuse_w/2),cg_z+(fuse_h/2)];
p3 = [cg_x+(fuse_l1-fuse_l2),cg_y-(fuse_w/2),cg_z+(fuse_h/2)];
p4 = [cg_x+(fuse_l1-fuse_l2),cg_y-(fuse_w/2),cg_z-(fuse_h/2)];
p5 = [cg_x+(fuse_l1-fuse_l2),cg_y+(fuse_w/2),cg_z-(fuse_h/2)];
%gövedenin kalanı
p6 = [cg_x+fuse_l1+fuse_l3,cg_y,cg_z];
%kanat
p7 = [cg_x+(fuse_l1),cg_y+(wing_w/2),cg_z+(wing_h)];
p8 = [cg_x+(fuse_l1+wing_l),cg_y+(wing_w/2),cg_z+(wing_h)];
p9 = [cg_x+(fuse_l1),cg_y-(wing_w/2),cg_z+(wing_h)];
p10 = [cg_x+(fuse_l1+wing_l),cg_y-(wing_w/2),cg_z+(wing_h)];
%kuyruk
p11 = [cg_x+(fuse_l1+fuse_l3-tail_l),cg_y+(tail_w/2),cg_z+tail_h];
p12 = [cg_x+(fuse_l1+fuse_l3),cg_y+(tail_w/2),cg_z+tail_h];
p13 = [cg_x+(fuse_l1+fuse_l3),cg_y-(tail_w/2),cg_z+tail_h];
p14 = [cg_x+(fuse_l1+fuse_l3-tail_l),cg_y-(tail_w/2),cg_z+tail_h];
%dikey kuyruk
p15 = [cg_x+(fuse_l1+fuse_l3-tail_l),cg_y,cg_z+(tail_h)];
p16 = [cg_x+fuse_l1+fuse_l3,cg_y,cg_z+(vertical_tail_h)];

points_array = [p1;p2;p3;p4;p5;p6;p7;p8;p9;p10;p11;p12;p13;p14;p15;p16];

outPoints = points_array;

end

%görselleştirme için2
function handle = VisualizeAircraft(points_array, i, handle)

    V = points_array;

    F = [... % Face definitions
        1, 2,  3;...  % burun1
        1, 4,  5;...  % burun2
        1, 3,  4;...  % burun3
        1, 5,  2;...  % burun4

        2, 3,  6;...  % gövde1
        3, 4,  6;...  % gövde2
        4, 5,  6;...  % gövde3
        5, 2,  6;...  % gövde4

        7, 8,  10;...  % kanat1
        7, 10, 9;...   % kanat2

        11, 12,  13;...  % yatay kuyruk1
        11, 13, 14;...   % yatay kuyruk2
        15, 16,  6;...   % dikey kuyruk
    ];

    % Define colors for each face
    myred = [1, 0, 0];
    mygreen = [0, 1, 0];
    myblue = [0, 0, 1];
    myyellow = [1, 1, 0];
    mycyan = [0, 1, 1];

    colors = [... 
        myred;...    % burun
        myred;...
        myred;...
        myred;...

        mygreen;...  % gövde
        mygreen;...
        mygreen;...
        mygreen;...

        myblue;...   % kanat
        myblue;...

        myyellow;... % yatay kuyruk
        myyellow;...
        mycyan;...   % dikey kuyruk
    ];

    if i == 0
        % İlk çizim
        handle = patch("Vertices", V, "Faces", F, "FaceVertexCData", colors, "FaceColor", "flat");
    else
        % Çizimi güncelle
        set(handle, "Vertices", V,"Faces",F);
        drawnow;
    end
end

%inertiadan body frame geçiş matrisi
function R_b_v = calc_R_b_v(phi,theta,psi)

R_b_v = [
    cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi);
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi);
   -sin(theta),          sin(phi)*cos(theta),                              cos(phi)*cos(theta)
];

end

%pozisyon değişimi
function p_dot = calc_ned_dot(u,w,v,phi,theta,psi,R_b_v)

p_dot = R_b_v * [u;w;v];

end

%hız değişimi
function velocity_dot = calc_velocity_dot(p,q,r,u,w,v,mass,fx,fy,fz)

A = [r*v-q*w;p*w-r*u;q*u-p*v];
f_xyz = [fx;fy;fz];
velocity_dot = A+((1/mass)*f_xyz);

end

%euler açıları değişmi
function euler_dot = calc_euler_dot(p,q,r,phi,theta,psi)

W = [
    1, sin(phi)*tan(theta), cos(phi)*tan(theta);
    0, cos(phi),           -sin(phi);
    0, sin(phi)/cos(theta), cos(phi)/cos(theta)
];

% Euler açı türevleri
euler_dot = W * [p; q; r];

end

%Açısal hız değişmi
function omega_dot = calc_omega_dot(J, p,q,r, l,m,n)
    % Girdi parametreleri:
    % J: Inersiya matrisi (3x3)

    % omega: Angular hız vektörü [p; q; r]
    omega = [p; q; r];
    % mb: Dışsal moment vektörü [l; m; n]
    mb = [l; m; n];
    % Angular momentum (hb = J * omega)
    hb = J * omega;

    % Angular momentum türevi: dh/dt = d(hb)/dt + omega x hb
    d_hb_dt = cross(omega, hb);  % omega x hb

    % Dönme hareketi denklemi: J * omega_dot + omega x (J * omega) = mb
    omega_dot = J \ (mb - d_hb_dt);  % omega_dot = inv(J) * (mb - omega x (J * omega))
end

%----------KUVVETLERIN HESABI-------------------

% Yerçekimi etkileri
function gavitional_force = calc_gravitional_force(mass,gravity,phi,theta,psi)
gavitional_force = [-mass*gravity*sin(theta);...
                     mass*gravity*cos(theta)*sin(phi);...
                     mass*gravity*cos(theta)*sin(phi)];
end

%aerodinamik kuvvetler ve momentler

%CX,CY,CZ hesabı aerodinamik katsayılar
function aerodynamic_conff = calc_aerodynamic_forces(CD,CL,CD_a,CL_a,CD_q,CL_q,CD_delta_e,CL_delta_e,alpha)

   CX_alpha = -CD_a*cos(alpha)+CL_a*sin(alpha);
   CX_qalpha = -CD_q*cos(alpha)+CL_q*sin(alpha);
   CX_deltaE_alpha = -CD_delta_e*cos(alpha)+CL_delta_e*sin(alpha);

   CZ_alpha = -CD_a*sin(alpha)-CL_a*cos(alpha);
   CZ_qalpha = -CD_q*sin(alpha)-CL_q*cos(alpha);
   CZ_deltaE_alpha = -CD_delta_e*sin(alpha)-CL_delta_e*cos(alpha);
   
   aerodynamic_conff = [CX_alpha; CX_qalpha; CX_deltaE_alpha; CZ_alpha; CZ_qalpha; CZ_deltaE_alpha];

end

% dikey etkiyen kuvvetlerin hesabı
function longitudal_aero_forces = calc_longitudal_aero_forces(ro,Va,S,c,p,q,r,b,delta_e,delta_a,delta_r,...
                                        CX_alpha, CX_qalpha, CX_deltaE_alpha,...
                                        CZ_alpha,CZ_qalpha,CZ_deltaE_alpha,...
                                        CY,CY_beta,CY_r,CY_delta_a,CY_delta_r,CY_p...
                                        )%moment
general_formula = 0.5*ro*Va*Va*S;

X = CX_alpha + (CX_qalpha*(c/(2*Va))*q)+CX_deltaE_alpha;
Y = CY+CY_beta+(CY_p*(b/(2*Va))*p)+(CY_r*(b/(2*Va))*r)+CY_delta_a*delta_a+CY_delta_r*delta_r;
Z = CZ_alpha+(CZ_qalpha*(c/(2*Va))*q)+CZ_deltaE_alpha*delta_e;

longitudal_aero_forces = [general_formula*X;...
                          general_formula*Y;...
                          general_formula*Z];
end

%itki hesabı
function longitudal_thrust_forces = calc_thrust_forces(rho,Va,S_prob,C_prob,K_motor,delta_throttle)
    thrust_force_x = 0.5*rho*S_prob*C_prob*(((K_motor*delta_throttle)^2)+(Va^2));
    longitudal_thrust_forces = [thrust_force_x;0;0];
end

%diket etkiyen net kuvvet
function longitudal_forces = calc_longitudal_forces(gavitional_force,longitudal_aero_forces,longitudal_thrust_forces)

longitudal_forces = gavitional_force + longitudal_aero_forces + longitudal_thrust_forces;
end

% momentlerin hesabı
function moment_aero_forces = calc_moment_aero_forces(rho,Va,S,c,b,p,q,beta,...
    Cl,Cl_beta,Cl_p,Cl_r,Cl_delta_a,delta_a,Cl_delta_r,delta_r,...
    alpha,Cm,Cm_a,Cm_q,Cm_delta_e,delta_e,...
    Cn,Cn_beta,Cn_p,Cn_r,Cn_delta_a,Cn_delta_r,r)

    general_formula = 0.5*rho*Va*Va*S;
    
    X = b*(Cl+Cl_beta*beta+(Cl_p*(b/(2*Va))*p)+(Cl_r*(b/(2*Va))*p)+Cl_delta_a*delta_a+Cl_delta_r*delta_r);
    Y = c*(Cm+Cm_a*alpha+(Cm_q*(c/(2*Va))*q)+Cm_delta_e*delta_e);
    Z = b*(Cn+Cn_beta*beta+(Cn_p*(b/(2*Va))*p)+(Cn_r*(b/(2*Va))*r)+Cn_delta_a*delta_a+Cn_delta_r*delta_r);

    moment_aero_forces = [X*general_formula;Y*general_formula;Z*general_formula];
end

function moment_thrust_forces = calc_moment_thrust_forces(ro,Va,S,K_motor,delta_throttle,K_omega)

    general_formula = 0.5*ro*Va*Va*S;

    X = -K_motor*((K_omega*delta_throttle)^2);

    moment_thrust_forces = [X*general_formula;0;0];

end

function moment_torque = calc_moment_torque(moment_aero_forces,moment_thrust_forces)

    moment_torque = moment_aero_forces + moment_thrust_forces;

end

%atmospheric

function Hs = calc_Hs(L_u,L_v,L_w,sigma_u,sigma_v,sigma_w,Va,s) %gürültü modeli

    Hu = sigma_u*((2*Va/Lu)^(1/2))*(1/(s+Va/L_u));
    
    Hv = sigma_v*((3*Va/L_v)^(1/2))*(s+(Va/(3^(1/2))*L_v)/((s+(Va/L_v))^2));
    
    Hw = sigma_w*((3*Va/L_w)^(1/2))*(s+(Va/(3^(1/2))*L_w)/((s+(Va/L_w))^2));
    
    Hs = [Hu;Hv;Hw];

end

function V_b_w = calc_V_b_w(R_b_v,Vws)

    V_b_w = R_b_v*Vws;

end

function Va_NED = calc_airspeed_NED(V_b_w,u,v,w) %hava hızı (airspeed)

    V_NED = [u;v;w];
    
    Va_NED = V_NED-V_b_w;

end

function Va = calc_airspeed(Va_NED) %hava hızı (airspeed)
    Va = ((Va_NED(1)^2)+(Va_NED(2)^2)+(Va_NED(3)^2))^(1/2);
end

function alpha = calc_alpha(Va_NED) % hücüm açısı (angle of atack)
    alpha = atan2(Va_NED(3),Va_NED(1));
end

function beta = calc_beta(Va_NED,Va) % kayma açısı (side slip angle)
    beta = asin(Va_NED(2)/Va);
end