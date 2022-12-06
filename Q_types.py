import simpy
import numpy as np
import random

class Queues():
    def __init__(self, arrival_intvl, service_intvl, n_serv, n_cust, FIFO=True,Det=False,LongT=False):
        self.FIFO = FIFO
        self.Det = Det
        self.LongT = LongT

        
        # model params
        self.arrival_intvl = arrival_intvl
        self.service_intvl = service_intvl
        self.n_cust = n_cust
        
        # Setting up SimPy Environment
        self.env = simpy.Environment()
        if self.FIFO == True:
            self.server = simpy.Resource(self.env, capacity = n_serv)
        else:
            self.server = simpy.PriorityResource(self.env, capacity=n_serv)
        

        # tracking metrics
        self.wait_time = np.zeros(n_cust)
        self.serv_time = np.zeros(n_cust)
        

    def sim(self):
        self.env.process(self.source())
        self.env.run()

        return self.wait_time, self.serv_time

    def source(self):
        """Generates customers randomly.
        Adapted from the SimPy Tutorial: https://simpy.readthedocs.io/en/latest/examples/bank_renege.html"""

        for i in range(self.n_cust):
            c = self.customer(i)
            self.env.process(c)
            t = random.expovariate(1/self.arrival_intvl)
            yield self.env.timeout(t)
            
        
    def customer(self, i):
        """Customer arrives, is served and leaves. 
        Adapted from the SimPy Tutorial: https://simpy.readthedocs.io/en/latest/examples/bank_renege.html"""
        arrive = self.env.now

        # Only necessary for LongTail:
        exp_mu = (self.service_intvl)* 0.5 # exponential mu parameter
        exp_mu_long = exp_mu * 5 # hyperexponential mu parameter

        # Decision Tree for all queueing types and distributions
        if self.FIFO == True:
            if self.Det == True: # Deterministic Service
                with self.server.request() as req:
                    # wait
                    yield req

                    # measure wait
                    self.wait_time[i] = self.env.now - arrive

                    # deterministic service time
                    yield self.env.timeout(self.service_intvl)

            else: # Exponential Distro
                if self.LongT == True:
                    with self.server.request() as req:
                        # wait
                        yield req

                        # measure wait time
                        self.wait_time[i] = self.env.now - arrive
                        
                        choice = np.random.uniform()
                        if choice <= 0.25:
                            service_t = random.expovariate(1/exp_mu_long) # Long-tailed with beta = fat-tailed rate 
                        else:
                            service_t = random.expovariate(1/exp_mu) # Normal with beta = normal-tailed rate

                        yield self.env.timeout(service_t)
                # Exponential Distribution
                else:
                    with self.server.request() as req:
                        # wait
                        yield req

                        # measure wait time
                        self.wait_time[i] = self.env.now - arrive
                        
                        service_t = random.expovariate(1.0 / self.service_intvl) # check if this is correct
                        yield self.env.timeout(service_t)
                        self.serv_time[i] = service_t
                

        # If queueing regime is SJF:
        else:
            # sample service time from distro
            service_t = random.expovariate(1/self.service_intvl)

            with self.server.request(priority=service_t) as req:
                yield req

                # measure wait time
                self.wait_time[i] = self.env.now - arrive

                yield self.env.timeout(service_t)

                # measure service time
                self.serv_time[i] = service_t
