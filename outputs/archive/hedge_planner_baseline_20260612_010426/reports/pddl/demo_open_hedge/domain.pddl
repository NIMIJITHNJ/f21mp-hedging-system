(define (domain hedge_plan_demo_open_hedge-domain)
 (:requirements :strips :negative-preconditions :disjunctive-preconditions :equality :numeric-fluents)
 (:predicates 
             (position_open)
             (margin_available)
             (rehedge_allowed)
             (roll_required)
             (roll_completed)
             (decision_completed)
             (hold_allowed)
 )
 (:functions 
             (current_hedge_bps)
             (target_hedge_bps)
             (max_hedge_bps)
             (estimated_trade_cost_micros)
             (max_trade_cost_micros)
 )
 (:action open_hedge
  :parameters ()
  :precondition (and (not (position_open)) (< 0 (target_hedge_bps)) (margin_available) (rehedge_allowed) (<= (estimated_trade_cost_micros) (max_trade_cost_micros)) (<= (target_hedge_bps) (max_hedge_bps)))
  :effect (and (position_open) (assign (current_hedge_bps) (target_hedge_bps)) (decision_completed)))
 (:action increase_hedge
  :parameters ()
  :precondition (and (position_open) (or (not (roll_required)) (roll_completed)) (< (current_hedge_bps) (target_hedge_bps)) (margin_available) (rehedge_allowed) (<= (estimated_trade_cost_micros) (max_trade_cost_micros)) (<= (target_hedge_bps) (max_hedge_bps)))
  :effect (and (assign (current_hedge_bps) (target_hedge_bps)) (decision_completed)))
 (:action reduce_hedge
  :parameters ()
  :precondition (and (position_open) (or (not (roll_required)) (roll_completed)) (< 0 (target_hedge_bps)) (< (target_hedge_bps) (current_hedge_bps)) (rehedge_allowed) (<= (estimated_trade_cost_micros) (max_trade_cost_micros)))
  :effect (and (assign (current_hedge_bps) (target_hedge_bps)) (decision_completed)))
 (:action close_hedge
  :parameters ()
  :precondition (and (position_open) (= (target_hedge_bps) 0) (rehedge_allowed) (<= (estimated_trade_cost_micros) (max_trade_cost_micros)))
  :effect (and (not (position_open)) (assign (current_hedge_bps) 0) (decision_completed)))
 (:action hold_position
  :parameters ()
  :precondition (and (hold_allowed))
  :effect (and (decision_completed)))
 (:action roll_futures_forward
  :parameters ()
  :precondition (and (position_open) (roll_required) (not (roll_completed)))
  :effect (and (roll_completed)))
)
