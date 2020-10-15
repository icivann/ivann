<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <DropdownInput @value-change="updateValue"
                   :value=value
                   :items="this.items"></DropdownInput>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import IntegerInc from '@/baklava/input/IntegerInc.vue';
import DropdownInput from '@/baklava/input/DropdownInput.vue';

@Component({
  components: { DropdownInput, IntegerInc },
})
export default class Dropdown extends Vue {
  @Prop() value!: string;

  @Prop({ type: String }) name!: string;

  @Prop({ type: Object })
  option!: { items: [string] };

  private items = this.option.items || [];

  updateValue(newValue: string) {
    this.$emit('input', newValue);
  }
}
</script>
