import dolfin
import matplotlib.pyplot as plt
import mshr
import numpy as np
import math

R_in = 1.0 # radius of the inclusion
R_out = 3.0 # radius of the outter matrix

E_m = 1.0 # Young's modulus for the matrix
nu_m = 0.3 # Poisson ratio for the matrix
E_i = 10.0 # Young's modulus for the inclusion
nu_i = 0.3 # Poisson ratio for the inclusion

h = 0.3*R_in # Size of elements
degreeFE = 1 # Degree of the finite elements

ONE = dolfin.Constant(1.)

MATRIX_ID = 1
INCLUSION_ID = 2

L_in = 2*np.pi*R_in # perimeter of the inclusion
L_out = 2*np.pi*R_out # perimeter of the matrix

N_in = int(L_in/h) # number of mesh points on the perimeter 
                   # of the inclusion
N_out = int(L_out/h) # number of mesh points on the perimeter 
                     # of the matrix

print(N_in)
print(N_out)

origin = dolfin.Point(0., 0.)

Omega_i = mshr.Circle(origin, R_in, segments=N_in)
Omega = mshr.Circle(origin, R_out, segments=N_out)

Omega.set_subdomain(MATRIX_ID, Omega-Omega_i) # we are putting tags in parts of the mesh
Omega.set_subdomain(INCLUSION_ID, Omega_i)    # we will use them later

mesh = mshr.generate_mesh(Omega, resolution=2*R_out/h)

dolfin.plot(mesh)
plt.show()
# we define a function = 1 in the matrix and = 2 in the inclusion
subdomain_data_2d = dolfin.MeshFunction("size_t", # the function returns a positive integer
                                        mesh, # it is define over the entire mesh
                                        dim=2, # the function is defined on the cells (not edges nor vertices)
                                        value=mesh.domains() # the function value is in fact
                                                             # given by the tag we have put while creating the mesh
                                       )

plt.colorbar(dolfin.plot(subdomain_data_2d)) # we plot this function, note the added color scale on the side
plt.show()
# we need to be able to integrate over the matrix only or the inclusion only
# so in addition of the classical dolfin measure dx, we define dx(1) and dx(2)
dx = dolfin.Measure("dx", domain=mesh, subdomain_data=subdomain_data_2d)

# area of the global mesh
dolfin.assemble(ONE*dx)

# area of the inclusion
dolfin.assemble(ONE*dx(INCLUSION_ID))

# area of the matrix
dolfin.assemble(ONE*dx(MATRIX_ID))

"""Debut de l'exercice"""

mu_mat=E_m/(2*(1+nu_m))
lamb_mat=(2*mu_mat*nu_m)/(1-2*nu_m)

mu_inc=E_i/(2*(1+nu_i))
lamb_inc=(2*mu_inc*nu_i)/(1-2*nu_i)

# sigma(eps)
def stress(eps, lamb, mu):
    return lamb*dolfin.tr(eps)*dolfin.Identity(mesh.geometric_dimension())+2*mu*eps

# eps(u)
def strain(u):
    return dolfin.sym(dolfin.grad(u))

element = dolfin.VectorElement('P', cell=mesh.ufl_cell(), degree=1, dim=mesh.geometric_dimension())
V = dolfin.FunctionSpace(mesh, element)

u = dolfin.TrialFunction(V)
v = dolfin.TestFunction(V)

sig_i=stress(strain(u),lamb_inc,mu_inc)
sig_m=stress(strain(u),lamb_mat,mu_mat)

eps_v=strain(v)
eps_u=strain(u)

# The system comprises two subdomains
# The total bilinear form is the addition of integrals on each subdomain
a_i = dolfin.inner(sig_i, eps_v)*dx(INCLUSION_ID)
a_m = dolfin.inner(sig_m, eps_v)*dx(MATRIX_ID)

bilinear_form = a_i + a_m

b= dolfin.Constant((0., 0.))
linear_form = dolfin.inner(b, v)*dolfin.dx

"""Boundry"""

tol = 1E-14

u_D = dolfin.Expression(('x[1]','x[0]'),degree=1)

def boundary(x,on_boundary):
    return on_boundary 
  
bc1 = dolfin.DirichletBC(V, u_D, boundary)

bc= [bc1]

A, p = dolfin.assemble_system(bilinear_form, linear_form, bc)
usol = dolfin.Function(V)
us_vector = usol.vector()
solver = dolfin.LUSolver("mumps")
solver.set_operator(A)
solver.solve(us_vector,p)
plt.colorbar(dolfin.plot(usol, mode='displacement',title=r"$\Vert u \Vert$",wireframe = True),orientation="vertical")
plt.show()
"""Compute the Solution to the elasticity problem"""

eps_sol = dolfin.sym(dolfin.grad(usol))
P1_tens = dolfin.TensorFunctionSpace(mesh, 'DG' ,0 )
strain_sol = dolfin.project(eps_sol,P1_tens)
dolfin.plot(strain_sol[0,0])
plt.title(r"$\varepsilon_{xx}$ - DG0 projection")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()

dolfin.plot(strain_sol[0,1])
plt.title(r"$\varepsilon_{xy}$ - DG0 projection")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()

dolfin.plot(strain_sol[1,1])
plt.title(r"$\varepsilon_{yy}$ - DG0 projection")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()

# Question 10
# Dans l'inclusion ils sont uniformes
# les composantes dominantes sont :

# Question 11
moy_epsxx_i = dolfin.assemble(eps_sol[0,0]*dx(INCLUSION_ID))/(dolfin.assemble(ONE*dx(INCLUSION_ID)))
moy_epsxy_i = dolfin.assemble(eps_sol[0,1]*dx(INCLUSION_ID))/(dolfin.assemble(ONE*dx(INCLUSION_ID)))
moy_epsyy_i = dolfin.assemble(eps_sol[1,1]*dx(INCLUSION_ID))/(dolfin.assemble(ONE*dx(INCLUSION_ID)))
print(moy_epsxx_i)
print(moy_epsxy_i)
print(moy_epsyy_i)
moy_epsxx_m = dolfin.assemble(eps_sol[0,0]*dx(MATRIX_ID))/(dolfin.assemble(ONE*dx(MATRIX_ID)))
moy_epsxy_m = dolfin.assemble(eps_sol[0,1]*dx(MATRIX_ID))/(dolfin.assemble(ONE*dx(MATRIX_ID)))
moy_epsyy_m = dolfin.assemble(eps_sol[1,1]*dx(MATRIX_ID))/(dolfin.assemble(ONE*dx(MATRIX_ID)))
print(moy_epsxx_m)
print(moy_epsxy_m)
print(moy_epsyy_m)
#Question 12
dev_i_xy = dolfin.assemble(abs(strain_sol[0,1] - moy_epsxy_i)*dx(INCLUSION_ID)) / moy_epsxy_i
print("dev = ",dev_i_xy)
#compatible avec le cisaillement pur

mu_m = E_m/(2*(1+nu_m))
mu_i = E_i/(2*(1+nu_i))
q = (3-4*nu_m)/(8*mu_m*(1-nu_m))
b = 1/(1+2*q*(mu_i-mu_m))
print('eps_xy_inclusion = ',b)

class EshelbyDisk:
    def __init__(self, γ, χ, ν_i, ν_m):
        self.γ = γ
        self.χ = χ
        self.ν_i = ν_i
        self.ν_m = ν_m

    def create_lhs(self):
        return np.array(
            [
                [2, 2 * self.γ ** 2, 2 * self.γ ** 4, 2 * self.γ ** 6, 0, 0],
                [
                    2 * self.ν_m * (self.ν_m - 1),
                    -(self.γ ** 2) * self.ν_m * (2 * self.ν_m - 1),
                    -2 * self.γ ** 4 * self.ν_m * (self.ν_m - 1),
                    self.γ ** 6 * (self.ν_m - 1) * (2 * self.ν_m - 3),
                    0,
                    0,
                ],
                [1, 1, 1, 1, -1, -1],
                [
                    2 * self.ν_i * self.ν_m * (self.ν_m - 1),
                    -self.ν_i * self.ν_m * (2 * self.ν_m - 1),
                    -2 * self.ν_i * self.ν_m * (self.ν_m - 1),
                    self.ν_i * (self.ν_m - 1) * (2 * self.ν_m - 3),
                    2 * self.ν_i * self.ν_m * (self.ν_m - 1),
                    -self.ν_m * (2 * self.ν_i - 3) * (self.ν_m - 1),
                ],
                [
                    3 * (self.ν_i + 1) * (self.ν_m - 1),
                    -self.ν_i - 1,
                    -(self.ν_i + 1) * (self.ν_m - 1),
                    0,
                    self.χ * (self.ν_m - 1) * (self.ν_m + 1),
                    0,
                ],
                [
                    -6 * self.ν_i * self.ν_m * (self.ν_i + 1) * (self.ν_m - 1),
                    self.ν_i * self.ν_m * (self.ν_i + 1),
                    -2 * self.ν_i * self.ν_m * (self.ν_i + 1) * (self.ν_m - 1),
                    -3 * self.ν_i * (self.ν_i + 1) * (self.ν_m - 1),
                    2 * self.χ * self.ν_i * self.ν_m * (self.ν_m - 1) * (self.ν_m + 1),
                    3 * self.χ * self.ν_m * (self.ν_m - 1) * (self.ν_m + 1),
                ],
            ]
        )

    def create_rhs(self):
        return np.array(
            [2 * self.γ ** 4, -2 * self.γ ** 4 * self.ν_m * (self.ν_m - 1), 0, 0, 0, 0]
        )

    def compute_integration_constants(self):
        return np.linalg.solve(self.create_lhs(), self.create_rhs())

    def to_expression(self, a=1.0, degree=4):
        A_m3, A_m1, A_1, A_3, C_1, C_3 = self.compute_integration_constants()
        ρ = dolfin.Expression("sqrt(pow(x[0], 2)+pow(x[1], 2))/a", degree=degree, a=a)
        θ = dolfin.Expression("atan2(x[1], x[0])", degree=degree)

        params = {
            "A_m3": A_m3,
            "A_m1": A_m1,
            "A_1": A_1,
            "A_3": A_3,
            "B_m3": -A_m3,
            "B_m1": (1 - 2 * self.ν_m) / 2 / (1 - self.ν_m) * A_m1,
            "B_1": A_1,
            "B_3": (3 - 2 * self.ν_m) / 2 / self.ν_m * A_3,
            "C_1": C_1,
            "C_3": C_3,
            "D_1": C_1,
            "D_3": (3 - 2 * self.ν_i) / 2 / self.ν_i * C_3,
            "rho": ρ,
            "theta": θ,
        }

        F = dolfin.Expression(
            "rho > 1 ? A_m3*pow(rho, -3)+A_m1/rho+A_1*rho+A_3*pow(rho, 3): C_1*rho+C_3*pow(rho, 3)",
            degree=degree,
            **params
        )
        G = dolfin.Expression(
            "rho > 1 ? B_m3*pow(rho, -3)+B_m1/rho+B_1*rho+B_3*pow(rho, 3): D_1*rho+D_3*pow(rho, 3)",
            degree=degree,
            **params
        )
        u_r = dolfin.Expression("F*sin(2*theta)", degree=degree, F=F, rho=ρ, theta=θ)
        u_θ = dolfin.Expression("G*cos(2*theta)", degree=degree, G=G, rho=ρ, theta=θ)
        u = dolfin.Expression(
            ("u_r*cos(theta)-u_theta*sin(theta)", "u_r*sin(theta)+u_theta*cos(theta)"),
            degree=degree,
            theta=θ,
            u_r=u_r,
            u_theta=u_θ,
        )
        return u

solution = EshelbyDisk(R_out/R_in, E_i/E_m, nu_i, nu_m)

u_ref = solution.to_expression(R_in)

V_ref = dolfin.VectorFunctionSpace(mesh, 'P', 1)
u_ref_num = dolfin.interpolate(u_ref, V_ref)
dolfin.plot(0.15*u_ref_num, mode="displacement")
plt.show()

error_L2 = dolfin.errornorm(u_ref,usol,"L2")
print("The L2 error norm of the numerical solution : error_L2 =", error_L2)

liste_x = np.linspace(-R_out, R_out, num=100)

u_formule = 0.0*liste_x

for k, x_k in enumerate(liste_x):
    u_formule[k] = u_ref([x_k,0.0])[1]  


plt.plot(liste_x, u_formule, label='Analytical Solution')
plt.title('Analytical Solution')
plt.ylabel("u[1](x[0],0)", color = 'blue', fontsize = 12 )
plt.xlabel("x[0]", color = 'blue', fontsize = 12)
plt.legend()

liste_x2 = np.linspace(-R_out, R_out, num=100)
u_formule2 = 0.0*liste_x2

for k, x_k in enumerate(liste_x2):
    u_formule2[k] = usol([x_k,0.0])[1] 

plt.plot(liste_x2, u_formule2, label='Numerical Solution')
plt.title('Solution')
plt.ylabel("u[1](x[0],0)", color = 'blue', fontsize = 12 )
plt.xlabel("x[0]", color = 'blue', fontsize = 12)
plt.legend()
plt.axis([-1.6,-1.3,-1.5,-0.8])
plt.legend(loc="upper left", borderaxespad=0.)
plt.show()
