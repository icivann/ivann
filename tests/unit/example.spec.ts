import { shallowMount } from '@vue/test-utils';
import HelloWorld from '@/components/HelloWorld.vue';
import { diffStringsUnified } from 'jest-diff';

describe('HelloWorld.vue', () => {
  it('renders props.msg when passed', () => {
    const msg = 'new message';
    const wrapper = shallowMount(HelloWorld, {
      propsData: { msg },
    });
    const diff = diffStringsUnified(msg, msg);
    console.log(diff);
    expect(wrapper.text()).toMatch(msg);
  });
});
