from choiceforge.activitysim_mode_choice import _component_phase


def test_mode_choice_components_are_attributed_to_their_qualification_phase():
    assert _component_phase("trip_mode_choice") == 17
    assert _component_phase("tour_mode_choice") == 33
    assert _component_phase("atwork_subtour_mode_choice") == 34
